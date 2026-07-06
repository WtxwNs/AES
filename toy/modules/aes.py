import math

import torch


class AESScheduler:
    """Adaptive Entropy Scheduling from Appendix C of Tracking Drift."""

    def __init__(
        self,
        q=0.9,
        beta=0.95,
        kappa=1.0,
        lambda_min=1e-4,
        lambda_max=1.0,
    ):
        self.q = q
        self.beta = beta
        self.kappa = kappa
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.ema = None
        self.accumulated = 0.0
        self.step = 0
        self.value = lambda_min

    def update(self, td_errors):
        with torch.no_grad():
            raw = torch.quantile(td_errors.detach().abs().float().flatten(), self.q).item()

        if self.ema is None:
            self.ema = raw
        else:
            self.ema = self.beta * self.ema + (1.0 - self.beta) * raw

        self.step += 1
        self.accumulated += self.ema
        scheduled = self.kappa * math.sqrt(max(self.accumulated, 0.0) / self.step)
        self.value = min(max(scheduled, self.lambda_min), self.lambda_max)
        return {
            "AES/raw_proxy": raw,
            "AES/smoothed_proxy": self.ema,
            "AES/accumulated_proxy": self.accumulated,
            "AES/lambda": self.value,
        }
