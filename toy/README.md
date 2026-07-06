# Toy Non-Stationary Multi-Goal Experiments

This folder contains the toy experiments for
`Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary
Reinforcement Learning`.

The toy environment uses a 2D multi-goal task with progress-aligned goal drift.
The AES variant schedules the entropy coefficient online from TD residuals.

## Drift Modes

- `steady`
- `abrupt`
- `linear`
- `periodic`
- `mixed`

## Smoke Test

```bash
python train.py config=multigoal/flow_aes.yaml steps=120 warmup_steps=20 eval_every=50 plot_every=100000 device=cpu
```

## Paired Baseline / AES Runs

```bash
python train.py config=multigoal/flow_nonstationary.yaml config.drift_pattern=abrupt config.description=baseline_abrupt seed=1
python train.py config=multigoal/flow_aes.yaml config.drift_pattern=abrupt config.description=aes_abrupt seed=1
```

Repeat over the five drift modes and seeds `{1, 2, 3, 4, 5}`.
