# skrl / Isaac Gym AES Experiments

This folder contains the Isaac Gym experiments for
`Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary
Reinforcement Learning`.

Implemented carriers:

- SAC
- Energy-based normalizing-flow policy carrier

The Isaac drift wrapper applies gravity scaling and task-exposed damping hooks.
Because OmniIsaacGymEnvs tasks expose physics parameters differently, unsupported
fields are left unchanged rather than failing the run.

## Environment

Use the Isaac Sim / OmniIsaacGymEnvs container setup required by skrl. Inside the
container, set:

```bash
alias PYTHON_PATH=/isaac-sim/python.sh
```

Install the local package:

```bash
cd /workspace/skrl
PYTHON_PATH -m pip install -e .["torch"]
PYTHON_PATH -m pip install ray[tune]
```

## Single Run

```bash
PYTHON_PATH aestd_train.py --algorithm flow --task Ant --use-nonstationary-env --drift-pattern abrupt --use-aes --output-dir runs_aes_isaac/flow/Ant/abrupt/aes/seed_1
```

## Batch Launch

```bash
PYTHON_PATH aestd_runs.py --dry-run
```
