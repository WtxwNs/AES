"""Best-effort Isaac Gym non-stationarity hooks for AES experiments."""

import numpy as np


class IsaacDriftWrapper:
    """Apply progress-aligned gravity/damping drift when the task exposes hooks.

    OmniIsaacGymEnvs tasks differ in how physics parameters are exposed. This
    wrapper intentionally tries common setter/attribute names and otherwise
    leaves the environment unchanged while preserving the training interface.
    """

    def __init__(
        self,
        env,
        drift_pattern="steady",
        total_timesteps=1_000_000,
        drift_fraction=0.2,
        gravity_min=0.8,
        gravity_max=1.2,
        damping_min=0.7,
        damping_max=1.3,
    ):
        self.env = env
        self.drift_pattern = drift_pattern
        self.total_timesteps = max(int(total_timesteps), 1)
        self.drift_fraction = float(drift_fraction)
        self.gravity_min = float(gravity_min)
        self.gravity_max = float(gravity_max)
        self.damping_min = float(damping_min)
        self.damping_max = float(damping_max)
        self.global_step = 0
        self._base_gravity = self._read_gravity()
        self._base_damping = self._read_damping()

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        self._apply_drift()
        return self.env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.global_step += 1
        self._apply_drift()
        return self.env.step(*args, **kwargs)

    def _scales(self):
        progress = np.clip(self.global_step / self.total_timesteps, 0.0, 1.0)
        if self.drift_pattern == "steady":
            return 1.0, 1.0
        segment = max(self.drift_fraction, 1e-6)
        local = (progress % segment) / segment
        if self.drift_pattern == "abrupt":
            jumps = int(progress / segment)
            if jumps == 0:
                return 1.0, 1.0
            rng = np.random.default_rng(jumps)
            return rng.uniform(self.gravity_min, self.gravity_max), rng.uniform(self.damping_min, self.damping_max)
        if self.drift_pattern == "linear":
            if progress < segment:
                return 1.0, 1.0
            return (
                self.gravity_min + (self.gravity_max - self.gravity_min) * local,
                self.damping_min + (self.damping_max - self.damping_min) * local,
            )
        if self.drift_pattern == "periodic":
            wave = 0.5 * (1.0 + np.sin(2.0 * np.pi * progress / segment))
            return (
                self.gravity_min + (self.gravity_max - self.gravity_min) * wave,
                self.damping_min + (self.damping_max - self.damping_min) * wave,
            )
        if self.drift_pattern == "mixed":
            jumps = int(progress / segment)
            if jumps == 0:
                return 1.0, 1.0
            abrupt = 1.0 if jumps % 2 == 0 else 0.0
            return (
                self.gravity_min + (self.gravity_max - self.gravity_min) * (0.5 * abrupt + 0.5 * local),
                self.damping_min + (self.damping_max - self.damping_min) * (0.5 * abrupt + 0.5 * local),
            )
        raise ValueError(f"Unknown drift_pattern: {self.drift_pattern}")

    def _apply_drift(self):
        gravity_scale, damping_scale = self._scales()
        self._write_gravity(gravity_scale)
        self._write_damping(damping_scale)

    def _read_gravity(self):
        for obj in (self.env, getattr(self.env, "task", None), getattr(self.env, "_task", None)):
            if obj is None:
                continue
            if hasattr(obj, "gravity"):
                return getattr(obj, "gravity")
        return None

    def _write_gravity(self, scale):
        gravity = self._base_gravity
        if gravity is None:
            return
        value = gravity * scale
        for obj in (self.env, getattr(self.env, "task", None), getattr(self.env, "_task", None)):
            if obj is None:
                continue
            if hasattr(obj, "set_gravity"):
                obj.set_gravity(value)
                return
            if hasattr(obj, "gravity"):
                setattr(obj, "gravity", value)
                return

    def _read_damping(self):
        for obj in (self.env, getattr(self.env, "task", None), getattr(self.env, "_task", None)):
            if obj is None:
                continue
            for name in ("joint_damping", "damping", "default_dof_damping"):
                if hasattr(obj, name):
                    return name, getattr(obj, name)
        return None

    def _write_damping(self, scale):
        if self._base_damping is None:
            return
        name, value = self._base_damping
        for obj in (self.env, getattr(self.env, "task", None), getattr(self.env, "_task", None)):
            if obj is not None and hasattr(obj, name):
                setattr(obj, name, value * scale)
                return


def maybe_wrap_isaac_drift(env, cfg):
    if not cfg.get("use_nonstationary_env", False):
        return env
    return IsaacDriftWrapper(
        env,
        drift_pattern=cfg.get("drift_pattern", "steady"),
        total_timesteps=cfg.get("timesteps", 1_000_000),
        drift_fraction=cfg.get("drift_fraction", 0.2),
        gravity_min=cfg.get("gravity_min", 0.8),
        gravity_max=cfg.get("gravity_max", 1.2),
        damping_min=cfg.get("damping_min", 0.7),
        damping_max=cfg.get("damping_max", 1.3),
    )
