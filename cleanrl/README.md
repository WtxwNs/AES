# CleanRL AES Experiments

This folder contains the MuJoCo continuous-control experiments for
`Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary
Reinforcement Learning`.

Implemented carriers:

- SAC
- PPO
- Energy-based normalizing-flow policy carrier

The non-stationary MuJoCo wrapper applies progress-aligned body-mass and
joint-friction drift.

## Install

```bash
pip install -r requirements-aes.txt
```

## Single Run

```bash
python cleanrl/sac_continuous_action.py --env-id Hopper-v4 --seed 1 --use-nonstationary-env --drift-pattern abrupt --use-aes --description runs_aes/sac/Hopper-v4/abrupt/aes/seed_1
```

## Batch Launch

Dry-run the full command matrix:

```bash
python aestd_runs.py --dry-run
```

Run selected MuJoCo jobs:

```bash
python aestd_runs.py --algorithms sac ppo flow --envs Hopper-v4 HalfCheetah-v4 Walker2d-v4 Ant-v4 Humanoid-v4
```

## Metrics

```bash
python aestd_metrics.py --runs-dir runs_aes --output aes_metrics.csv
```
