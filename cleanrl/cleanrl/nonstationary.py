"""Non-stationary environment wrappers for AES experiments."""

import numpy as np
import gymnasium as gym


class NonStationaryMujocoWrapper(gym.Wrapper):
    """Applies progress-aligned physics drift to MuJoCo environments."""

    def __init__(
        self,
        env,
        drift_pattern="steady",
        total_timesteps=1_000_000,
        drift_fraction=0.2,
        mass_min=0.7,
        mass_max=1.3,
        friction_min=0.5,
        friction_max=1.5,
    ):
        super().__init__(env)
        self.drift_pattern = drift_pattern
        self.total_timesteps = max(int(total_timesteps), 1)
        self.drift_fraction = float(drift_fraction)
        self.mass_min = float(mass_min)
        self.mass_max = float(mass_max)
        self.friction_min = float(friction_min)
        self.friction_max = float(friction_max)
        self.global_step = 0
        model = self.unwrapped.model
        self.base_body_mass = np.array(model.body_mass, copy=True) if hasattr(model, "body_mass") else None
        self.base_friction = np.array(model.dof_frictionloss, copy=True) if hasattr(model, "dof_frictionloss") else None

    def set_global_step(self, step):
        self.global_step = int(step)
        self._apply_drift()

    def reset(self, **kwargs):
        self._apply_drift()
        return self.env.reset(**kwargs)

    def step(self, action):
        self._apply_drift()
        return self.env.step(action)

    def _apply_drift(self):
        mass_scale, friction_scale = self._scales()
        model = self.unwrapped.model
        if self.base_body_mass is not None:
            model.body_mass[:] = self.base_body_mass * mass_scale
        if self.base_friction is not None:
            model.dof_frictionloss[:] = self.base_friction * friction_scale

    def _scales(self):
        progress = np.clip(self.global_step / self.total_timesteps, 0.0, 1.0)
        if self.drift_pattern == "steady":
            return 1.0, 1.0

        segment = max(self.drift_fraction, 1e-6)
        jumps = min(int(np.floor(progress / segment + 1e-10)), int(np.ceil(1 / segment)) - 1)
        local = float(np.clip(progress / segment - jumps, 0., 1.))

        if self.drift_pattern == "abrupt":
            if jumps == 0:
                return 1.0, 1.0
            rng = np.random.default_rng(jumps)
            return (
                rng.uniform(self.mass_min, self.mass_max),
                rng.uniform(self.friction_min, self.friction_max),
            )

        if self.drift_pattern == "linear":
            if progress < segment:
                return 1.0, 1.0
            return (
                self.mass_min + (self.mass_max - self.mass_min) * local,
                self.friction_min + (self.friction_max - self.friction_min) * local,
            )

        if self.drift_pattern == "periodic":
            wave = 0.5 * (1.0 + np.sin(2.0 * np.pi * progress / segment))
            return (
                self.mass_min + (self.mass_max - self.mass_min) * wave,
                self.friction_min + (self.friction_max - self.friction_min) * wave,
            )

        if self.drift_pattern == "mixed":
            if jumps == 0:
                return 1.0, 1.0
            abrupt = 1.0 if jumps % 2 == 0 else 0.0
            linear = local
            return (
                self.mass_min + (self.mass_max - self.mass_min) * (0.5 * abrupt + 0.5 * linear),
                self.friction_min + (self.friction_max - self.friction_min) * (0.5 * abrupt + 0.5 * linear),
            )

        raise ValueError(f"Unknown drift_pattern: {self.drift_pattern}")


def maybe_wrap_nonstationary(env, args):
    if not getattr(args, "use_nonstationary_env", False):
        return env
    if hasattr(env.unwrapped, "_goal_shift"):
        return env
    return NonStationaryMujocoWrapper(
        env,
        drift_pattern=args.drift_pattern,
        total_timesteps=args.total_timesteps,
        drift_fraction=args.drift_fraction,
        mass_min=args.mass_min,
        mass_max=args.mass_max,
        friction_min=args.friction_min,
        friction_max=args.friction_max,
    )


def set_global_step(envs, step):
    for env in envs.envs:
        current = env
        while isinstance(current, gym.Wrapper):
            if isinstance(current, NonStationaryMujocoWrapper):
                current.set_global_step(step)
                break
            current = current.env
        if hasattr(current.unwrapped, "_goal_shift"):
            current.unwrapped.set_global_step(step)


def make_task(env_id, args=None, **kwargs):
    """Use the same Toy environment for every carrier, without global registration."""
    if env_id != "NonStationaryMultiGoal-v0":
        return gym.make(env_id, **kwargs)
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "toy/toy_envs/multi_goal.py"
    spec = importlib.util.spec_from_file_location("aes_multi_goal", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NonStationaryMultiGoal(
        drift_pattern=args.drift_pattern if args and args.use_nonstationary_env else "steady",
        total_steps=args.total_timesteps if args else 1_000_000,
        drift_fraction=args.drift_fraction if args else .2)
