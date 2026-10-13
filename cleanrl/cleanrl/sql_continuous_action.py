"""PyTorch SQL + AES reference carrier (new reconstruction, not original run code).

Soft value uses uniform importance sampling of the bounded action integral.
The stochastic policy is trained by amortized SVGD toward exp(Q / alpha),
including the tanh change-of-variables term. This is SQL, not a renamed SAC.
Reference: Haarnoja et al., ICML 2017, https://proceedings.mlr.press/v70/haarnoja17a.html
The released AES artifacts do not specify the original SQL particle count;
these explicit defaults therefore require a new experimental validation.
"""
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from aes import AESConfig, AESScheduler
from sac_continuous_action import Args as SACArgs, SoftQNetwork, make_env
from nonstationary import set_global_step
from reproduction import write_run_config, log_evaluation, save_training_state


@dataclass
class Args(SACArgs):
    exp_name: str = "sql_continuous_action"
    autotune: bool = False
    value_particles: int = 64
    kernel_particles: int = 16
    policy_lr: float = 3e-4
    alpha: float = 1.0
    policy_frequency: int = 1


class ParticlePolicy(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        self.action_dim = int(np.prod(env.single_action_space.shape))
        self.net = nn.Sequential(nn.Linear(obs_dim + self.action_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, self.action_dim))
        self.register_buffer("scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2))
        self.register_buffer("bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2))

    def particles(self, obs, count, noise=None):
        if noise is None:
            noise = torch.randn(len(obs), count, self.action_dim, device=obs.device)
        states = obs[:, None].expand(-1, count, -1)
        latent = self.net(torch.cat((states, noise), dim=-1))
        return latent, latent.tanh() * self.scale + self.bias

    def act(self, obs, deterministic=False):
        if deterministic:
            # Fixed antithetic noise bank defines a reproducible approximation
            # to the mean of this implicit policy (no training RNG consumed).
            generator = torch.Generator(device=obs.device).manual_seed(0)
            half = torch.randn(1, 32, self.action_dim, generator=generator, device=obs.device)
            noise = torch.cat((half, -half), dim=1).expand(len(obs), -1, -1)
            return self.particles(obs, 64, noise)[1].mean(dim=1)
        return self.particles(obs, 1)[1][:, 0]


def soft_value(q_samples, alpha, log_volume):
    if alpha <= 0:
        raise ValueError("SQL temperature must be positive")
    return alpha * (torch.logsumexp(q_samples / alpha, dim=1) - math.log(q_samples.shape[1]) + log_volume)


def stein_direction(latent, score):
    """SVGD direction with source-particle gradient of an RBF kernel."""
    count = latent.shape[1]
    diff = latent[:, :, None, :] - latent[:, None, :, :]  # source j - destination i
    dist = diff.square().sum(-1)
    off_diagonal = ~torch.eye(count, dtype=torch.bool, device=latent.device)
    bandwidth = (dist[:, off_diagonal].median(-1).values / math.log(count + 1)).clamp_min(1e-3)
    kernel = (-dist / bandwidth[:, None, None]).exp()
    attraction = kernel[..., None] * score[:, :, None, :]
    repulsion = -2 * diff * kernel[..., None] / bandwidth[:, None, None, None]
    return (attraction + repulsion).mean(dim=1)


@torch.no_grad()
def evaluate(env, policy, device):
    obs, _ = env.reset(seed=list(range(env.num_envs)))
    done = np.zeros(env.num_envs, dtype=bool)
    returns = np.zeros(env.num_envs)
    while not done.all():
        actions = policy.act(torch.as_tensor(obs, dtype=torch.float32, device=device), True)
        obs, reward, terminated, truncated, _ = env.step(actions.cpu().numpy())
        returns += reward * ~done
        done |= terminated | truncated
    return float(returns.mean())


def train(args):
    if args.autotune:
        raise ValueError("The paper SQL baseline uses fixed alpha, not automatic tuning")
    if args.kernel_particles < 2 or args.value_particles < 1:
        raise ValueError("At least two kernel particles and one value particle are required")
    args.description = args.description or f"runs/sql/{args.env_id}/seed_{args.seed}"
    write_run_config(args.description, args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    env = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, False, "sql", args)])
    evaluation = gym.vector.SyncVectorEnv([make_env(args.env_id, i, i, False, "sql", args) for i in range(10)])
    policy = ParticlePolicy(env).to(device)
    critic = SoftQNetwork(env).to(device)
    target = SoftQNetwork(env).to(device)
    target.load_state_dict(critic.state_dict())
    q_optimizer = torch.optim.Adam(critic.parameters(), lr=args.q_lr)
    p_optimizer = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    aes = AESScheduler(AESConfig(args.aes_q, args.aes_beta, args.aes_kappa, args.aes_lambda_min, args.aes_lambda_max)) if args.use_aes else None
    alpha = aes.value if aes else args.alpha
    replay = ReplayBuffer(args.buffer_size, env.single_observation_space, env.single_action_space, device, handle_timeout_termination=False)
    writer = SummaryWriter(args.description)
    log_volume = torch.log(2 * policy.scale).sum()
    obs, _ = env.reset(seed=args.seed)
    if args.eval_frequency > 0:
        log_evaluation(writer, args.description, 0, evaluate(evaluation, policy, device))
    for step in range(args.total_timesteps):
        set_global_step(env, step)
        with torch.no_grad():
            actions = np.array([env.single_action_space.sample()]) if step < args.learning_starts else policy.act(torch.as_tensor(obs, device=device)).cpu().numpy()
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        actual_next = next_obs.copy()
        for i, trunc in enumerate(truncated):
            if trunc:
                actual_next[i] = infos["final_observation"][i]
        replay.add(obs, actual_next, actions, rewards, terminated, infos)
        obs = next_obs
        if step >= args.learning_starts:
            batch = replay.sample(args.batch_size)
            with torch.no_grad():
                count = args.value_particles
                states = batch.next_observations[:, None].expand(-1, count, -1).reshape(-1, obs.shape[1])
                candidates = (2 * torch.rand(args.batch_size, count, policy.action_dim, device=device) - 1) * policy.scale + policy.bias
                target_q = target(states, candidates.reshape(-1, policy.action_dim)).reshape(args.batch_size, count)
                backup = batch.rewards.flatten() + args.gamma * (1 - batch.dones.flatten()) * soft_value(target_q, alpha, log_volume)
            q = critic(batch.observations, batch.actions).flatten()
            if aes:
                alpha = aes.update(q.detach() - backup)
                with torch.no_grad():
                    backup = batch.rewards.flatten() + args.gamma * (1 - batch.dones.flatten()) * soft_value(target_q, alpha, log_volume)
            q_loss = F.mse_loss(q, backup)
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()
            if step % args.policy_frequency == 0:
                latent, particles = policy.particles(batch.observations, args.kernel_particles)
                source = latent.detach().requires_grad_()
                repeated_obs = batch.observations[:, None].expand(-1, args.kernel_particles, -1).reshape(-1, obs.shape[1])
                energy = critic(repeated_obs, (source.tanh() * policy.scale + policy.bias).reshape(-1, policy.action_dim)).reshape(args.batch_size, -1) / alpha
                log_jacobian = (2 * (math.log(2) - source - F.softplus(-2 * source))).sum(-1)
                score = torch.autograd.grad((energy + log_jacobian).sum(), source)[0]
                direction = stein_direction(source.detach(), score.detach()).detach()
                p_loss = -(latent * direction).sum(-1).mean()
                p_optimizer.zero_grad()
                p_loss.backward()
                p_optimizer.step()
            if step % args.target_network_frequency == 0:
                with torch.no_grad():
                    for param, target_param in zip(critic.parameters(), target.parameters()):
                        target_param.lerp_(param, args.tau)
            writer.add_scalar("losses/q_loss", q_loss.item(), step + 1)
            writer.add_scalar("losses/alpha", alpha, step + 1)
        completed = step + 1
        if args.eval_frequency > 0 and (completed % args.eval_frequency == 0 or completed == args.total_timesteps):
            set_global_step(evaluation, completed)
            log_evaluation(writer, args.description, completed, evaluate(evaluation, policy, device))
    save_training_state(args.description, model=policy.state_dict(), critic=critic.state_dict(), target=target.state_dict(),
                        aes=None if aes is None else aes.state_dict(), args=vars(args), step=args.total_timesteps)
    env.close()
    evaluation.close()
    writer.close()


if __name__ == "__main__":
    train(tyro.cli(Args))
