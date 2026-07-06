"""Template adapter for adding AES to an external Soft Q-Learning codebase.

The local repository does not vendor a SQL implementation. To reproduce the
SQL rows from the AES paper, copy this scheduler into the SQL update loop and
replace the static temperature in the soft Bellman target/value computation.
"""

from dataclasses import dataclass
import math

import torch


@dataclass
class AESConfig:
    q: float = 0.9
    beta: float = 0.95
    kappa: float = 1.0
    lambda_min: float = 1e-4
    lambda_max: float = 1.0


class AESScheduler:
    def __init__(self, config=AESConfig()):
        self.config = config
        self.ema = None
        self.accumulated = 0.0
        self.step = 0
        self.value = config.lambda_min

    def update_from_td_errors(self, *td_error_tensors):
        residuals = torch.cat([x.detach().abs().float().flatten() for x in td_error_tensors])
        raw = torch.quantile(residuals, self.config.q).item()
        self.ema = raw if self.ema is None else self.config.beta * self.ema + (1.0 - self.config.beta) * raw
        self.step += 1
        self.accumulated += self.ema
        scheduled = self.config.kappa * math.sqrt(max(self.accumulated, 0.0) / self.step)
        self.value = min(max(scheduled, self.config.lambda_min), self.config.lambda_max)
        return self.value


def sql_update_pseudocode(batch, q_network, target_q_network, optimizer, scheduler, gamma):
    """Non-runnable reference showing the required SQL insertion point."""
    q_pred = q_network(batch.states, batch.actions)
    with torch.no_grad():
        # Use the current scheduled value in the soft value / log-sum-exp term.
        soft_value = scheduler.value * target_q_network.soft_value(batch.next_states, temperature=scheduler.value)
        target = batch.rewards + (1.0 - batch.dones) * gamma * soft_value

    # Update AES from the observable TD residual and then rebuild the target
    # with the same scheduled value used by this gradient step.
    alpha = scheduler.update_from_td_errors(q_pred - target)
    with torch.no_grad():
        soft_value = alpha * target_q_network.soft_value(batch.next_states, temperature=alpha)
        target = batch.rewards + (1.0 - batch.dones) * gamma * soft_value

    loss = torch.nn.functional.mse_loss(q_pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": loss.item(), "alpha": alpha}
