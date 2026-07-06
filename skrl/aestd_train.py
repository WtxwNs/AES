import argparse

from skrl.resources.preprocessors.torch import RunningStandardScaler

from trainer_flow import _train as train_flow
from trainer_sac import _train as train_sac
from skrl.agents.torch.flow import FLOW_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC_DEFAULT_CONFIG


def base_cfg(args):
    cfg = (FLOW_DEFAULT_CONFIG if args.algorithm == "flow" else SAC_DEFAULT_CONFIG).copy()
    cfg["task_name"] = args.task
    cfg["batch_size"] = args.batch_size
    cfg["num_envs"] = args.num_envs
    cfg["timesteps"] = args.timesteps
    cfg["random_timesteps"] = args.random_timesteps
    cfg["learning_starts"] = args.learning_starts
    cfg["memory_size"] = args.memory_size
    cfg["gradient_steps"] = args.gradient_steps
    cfg["discount_factor"] = args.discount_factor
    cfg["state_preprocessor"] = RunningStandardScaler if args.state_preprocessor else None
    cfg["experiment"]["directory"] = args.output_dir
    cfg["experiment"]["write_interval"] = args.write_interval
    cfg["experiment"]["checkpoint_interval"] = args.checkpoint_interval
    cfg["use_nonstationary_env"] = args.use_nonstationary_env
    cfg["drift_pattern"] = args.drift_pattern
    cfg["drift_fraction"] = args.drift_fraction
    cfg["gravity_min"] = args.gravity_min
    cfg["gravity_max"] = args.gravity_max
    cfg["damping_min"] = args.damping_min
    cfg["damping_max"] = args.damping_max
    cfg["use_aes"] = args.use_aes
    cfg["aes_q"] = args.aes_q
    cfg["aes_beta"] = args.aes_beta
    cfg["aes_kappa"] = args.aes_kappa
    cfg["aes_lambda_min"] = args.aes_lambda_min
    cfg["aes_lambda_max"] = args.aes_lambda_max
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["sac", "flow"], required=True)
    parser.add_argument("--task", default="AllegroHand")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--memory-size", type=int, default=15000)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--random-timesteps", type=int, default=100)
    parser.add_argument("--learning-starts", type=int, default=100)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--write-interval", type=int, default=5000)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000_000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--state-preprocessor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-aes", action="store_true")
    parser.add_argument("--aes-q", type=float, default=0.9)
    parser.add_argument("--aes-beta", type=float, default=0.95)
    parser.add_argument("--aes-kappa", type=float, default=1.0)
    parser.add_argument("--aes-lambda-min", type=float, default=1e-4)
    parser.add_argument("--aes-lambda-max", type=float, default=1.0)
    parser.add_argument("--use-nonstationary-env", action="store_true")
    parser.add_argument("--drift-pattern", default="steady", choices=["steady", "abrupt", "linear", "periodic", "mixed"])
    parser.add_argument("--drift-fraction", type=float, default=0.2)
    parser.add_argument("--gravity-min", type=float, default=0.8)
    parser.add_argument("--gravity-max", type=float, default=1.2)
    parser.add_argument("--damping-min", type=float, default=0.7)
    parser.add_argument("--damping-max", type=float, default=1.3)
    parser.add_argument("--sac-alpha", type=float, default=0.1)
    parser.add_argument("--flow-alpha", type=float, default=0.1)
    parser.add_argument("--sigma-max", type=float, default=-0.3)
    parser.add_argument("--sigma-min", type=float, default=-5.0)
    args = parser.parse_args()

    cfg = base_cfg(args)
    if args.algorithm == "sac":
        cfg["learn_entropy"] = False
        cfg["initial_entropy_value"] = args.sac_alpha
        cfg["actor_learning_rate"] = 3e-4
        cfg["critic_learning_rate"] = 3e-4
        train_sac(cfg)
    else:
        cfg["entropy_value"] = args.flow_alpha
        cfg["learning_rate"] = 1e-3
        cfg["grad_norm_clip"] = 30
        cfg["sigma_max"] = args.sigma_max
        cfg["sigma_min"] = args.sigma_min
        train_flow(cfg)


if __name__ == "__main__":
    main()
