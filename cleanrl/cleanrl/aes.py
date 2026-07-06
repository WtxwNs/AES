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
        self.config = config
        self.ema = None
        self.accumulated = 0.0
        self.step = 0
        self.value = config.lambda_min

    def update(self, residuals):
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

    def scalars(self):
        return {
            "aes/raw_proxy_ema": 0.0 if self.ema is None else self.ema,
            "aes/accumulated_proxy": self.accumulated,
            "aes/lambda": self.value,
        }
