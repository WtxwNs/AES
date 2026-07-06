# AES release guide

This repository is prepared as a runnable code release for:

`Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning`

## Implemented carriers

- Toy flow-based carrier on `NonStationaryMultiGoal-v0`
- CleanRL SAC on MuJoCo continuous control
- CleanRL PPO on MuJoCo continuous control
- CleanRL flow-based carrier on MuJoCo continuous control
- skrl SAC on Omniverse Isaac Gym
- skrl flow-based carrier on Omniverse Isaac Gym

The local repository does not contain a SQL implementation. SQL is therefore
provided as an external adapter template in `external_sql_aes_adapter.py`.

## Core files

- `cleanrl/cleanrl/aes.py`: shared AES scheduler
- `cleanrl/cleanrl/nonstationary.py`: MuJoCo dynamics drift wrapper
- `cleanrl/cleanrl/sac_continuous_action.py`: SAC + AES
- `cleanrl/cleanrl/ppo_continuous_action.py`: PPO + AES
- `cleanrl/cleanrl/flow_continuous_action.py`: flow-based carrier + AES
- `toy/modules/aes.py`: toy AES scheduler
- `toy/toy_envs/multi_goal.py`: non-stationary toy goal drift
- `cleanrl/aestd_runs.py`: batch launcher
- `cleanrl/aestd_metrics.py`: TensorBoard metric aggregator
- `skrl/aestd_train.py`: single Isaac Gym AES run
- `skrl/aestd_runs.py`: Isaac Gym batch launcher
- `external_sql_aes_adapter.py`: SQL integration template
- `AES_EXPERIMENT_MATRIX.md`: full experiment-to-entrypoint mapping

## Smoke tests

Install the CleanRL AES dependencies:

```bash
cd cleanrl
pip install -r requirements-aes.txt
```

Toy:

```bash
cd toy
python train.py config=multigoal/flow_aes.yaml steps=120 warmup_steps=20 eval_every=50 plot_every=100000 device=cpu
```

CleanRL SAC on a lightweight environment:

```bash
cd cleanrl
python cleanrl/sac_continuous_action.py --env-id Pendulum-v1 --total-timesteps 200 --learning-starts 10 --batch-size 8 --use-aes --no-cuda --description runs_smoke/sac_aes
```

## MuJoCo reproduction

Dry-run all commands:

```bash
cd cleanrl
python aestd_runs.py --dry-run --total-timesteps 1000000
```

Run a single paired setting:

```bash
python cleanrl/sac_continuous_action.py --env-id Hopper-v4 --seed 1 --total-timesteps 1000000 --use-nonstationary-env --drift-pattern abrupt --description runs_aes/sac/Hopper-v4/abrupt/baseline/seed_1
python cleanrl/sac_continuous_action.py --env-id Hopper-v4 --seed 1 --total-timesteps 1000000 --use-nonstationary-env --drift-pattern abrupt --use-aes --description runs_aes/sac/Hopper-v4/abrupt/aes/seed_1
```

Aggregate TensorBoard metrics:

```bash
python aestd_metrics.py --runs-dir runs_aes --output aes_metrics.csv
```

## Isaac Gym reproduction

Inside the Isaac Sim container described in `skrl/README.md`:

```bash
cd /workspace/skrl
PYTHON_PATH aestd_runs.py --dry-run
PYTHON_PATH aestd_train.py --algorithm flow --task Ant --use-nonstationary-env --drift-pattern abrupt --use-aes --output-dir runs_aes_isaac/flow/Ant/abrupt/aes/seed_1
```

The Isaac drift wrapper applies gravity scaling and task-exposed damping hooks.
Because OmniIsaacGymEnvs tasks expose physics parameters differently, unsupported
fields are left unchanged rather than failing the run.

## Reported AES hyperparameters

- q: `0.9`
- EMA beta: `0.95`
- kappa: `1.0`
- off-policy lambda range: `[1e-4, 1.0]`
- PPO lambda range: `[1e-4, 0.1]`

## Release checklist

- Run toy smoke test.
- Run CleanRL smoke tests for SAC/PPO/flow-based carrier.
- Run at least one MuJoCo seed with baseline and AES.
- Dry-run `skrl/aestd_runs.py` in the Isaac container.
- Confirm `aestd_metrics.py` reads the produced TensorBoard logs.
- Add generated results separately; do not commit large checkpoints by default.
