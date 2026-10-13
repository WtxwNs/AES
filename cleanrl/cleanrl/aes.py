"""Adaptive Entropy Scheduling utilities.

Implements the scheduler described in Appendix C of
"Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary
Reinforcement Learning".
"""

import math
from dataclasses import dataclass

import torch


@dataclass
class AESConfig:
    q: float = 0.9
    beta: float = 0.95
    kappa: float = 1.0
    lambda_min: float = 1e-4
    lambda_max: float = 1.0


class AESScheduler:
    def __init__(self, config: AESConfig):
        if not (0 <= config.q <= 1 and 0 <= config.beta < 1 and config.kappa > 0
                and 0 < config.lambda_min <= config.lambda_max
                and all(math.isfinite(v) for v in vars(config).values())):
            raise ValueError("Invalid AES configuration")
        self.config = config
        self.ema = None
        self.accumulated = 0.0
        self.step = 0
        self.value = config.lambda_min

    def update(self, residuals):
        if residuals.numel() == 0 or not torch.isfinite(residuals).all():
            raise ValueError("AES requires nonempty finite TD residuals")
        with torch.no_grad():
            raw = torch.quantile(residuals.detach().abs().float().flatten(), self.config.q).item()

        if self.ema is None:
            self.ema = raw
        else:
            self.ema = self.config.beta * self.ema + (1.0 - self.config.beta) * raw

        self.step += 1
        self.accumulated += self.ema
        scheduled = self.config.kappa * math.sqrt(max(self.accumulated, 0.0) / self.step)
        self.value = min(max(scheduled, self.config.lambda_min), self.config.lambda_max)
        return self.value

    def state_dict(self):
        return {"config": vars(self.config).copy(), "ema": self.ema,
                "accumulated": self.accumulated, "step": self.step, "value": self.value}

    def load_state_dict(self, state):
        if state["config"] != vars(self.config):
            raise ValueError("AES checkpoint configuration differs")
        for key in ("ema", "accumulated", "step", "value"):
            setattr(self, key, state[key])

    def scalars(self):
        return {
            "aes/raw_proxy_ema": 0.0 if self.ema is None else self.ema,
            "aes/accumulated_proxy": self.accumulated,
            "aes/lambda": self.value,
        }
