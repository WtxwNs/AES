# Tracking Drift AES toy reproduction

This directory now contains a focused reproduction slice for the paper
`Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary
Reinforcement Learning`, using the existing flow-based toy infrastructure as
the carrier.

## What is implemented

- `NonStationaryMultiGoal-v0` with the Appendix C drift protocols:
  `steady`, `abrupt`, `linear`, `periodic`, and `mixed`.
- AES scheduler:
  `lambda_t = clip(lambda_min, lambda_max, kappa * sqrt(A_t / t))`.
- TD-error drift proxy for the flow-based carrier:
  quantile of minibatch absolute TD residuals, EMA smoothing, and accumulated
  proxy.
- TensorBoard logging under `AES/*`.

## Smoke test

```bash
python train.py config=multigoal/flow_aes.yaml steps=120 warmup_steps=20 eval_every=50 plot_every=100000 device=cpu
```

## Full toy runs

Run the baseline and AES variant with the same drift pattern and seed:

```bash
python train.py config=multigoal/flow_nonstationary.yaml config.drift_pattern=abrupt config.description=baseline_abrupt seed=1
python train.py config=multigoal/flow_aes.yaml config.drift_pattern=abrupt config.description=aes_abrupt seed=1
```

Repeat for:

```text
steady abrupt linear periodic mixed
```

and seeds:

```text
1 2 3 4 5
```

The paper evaluates every 10,000 environment steps on large tasks. The toy
configuration in this repository defaults to much shorter runs, so increase
`steps` and align `eval_every` as needed for paper-scale reporting.
