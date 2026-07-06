# Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning

This repository contains the code release for the ICML 2026 paper:

**Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning**

Tongxi Wang, Zhuoyang Xia, Xinran Chen, and Shan Liu.

Adaptive Entropy Scheduling (AES) is a lightweight plug-in mechanism for
non-stationary maximum-entropy reinforcement learning. AES uses online
TD-style residual proxies to schedule the entropy coefficient according to the
observed drift magnitude, increasing exploration after drift and relaxing it
during stable phases.

## What is Included

- Toy 2D non-stationary multi-goal experiments.
- MuJoCo continuous-control experiments with CleanRL SAC, PPO, and an
  energy-based normalizing-flow policy carrier.
- Isaac Gym experiments with skrl SAC and the same flow-based policy carrier.
- Non-stationary drift wrappers for goal, dynamics, and physics perturbations.
- Batch launchers, metric aggregation utilities, and release checks.
- SQL AES adapter template for external Soft Q-Learning implementations.

The local repository does not vendor a runnable SQL implementation. SQL support
is provided as an integration template in `external_sql_aes_adapter.py`.

## Repository Structure

- `toy/`: Toy 2D multi-goal AES experiments.
- `cleanrl/`: MuJoCo AES experiments using CleanRL SAC, PPO, and the flow-based
  carrier.
- `skrl/`: Isaac Gym AES experiments using skrl SAC and the flow-based carrier.
- `plot/`: Existing plotting utilities and reference result plotting scripts.
- `AES_RELEASE.md`: Release checklist, smoke tests, and reproduction commands.
- `AES_EXPERIMENT_MATRIX.md`: Mapping from paper experiments to code entry points.
- `external_sql_aes_adapter.py`: SQL integration template.
- `scripts/check_release.py`: Static release check used by CI.

## Quick Checks

Run the static release check:

```bash
python scripts/check_release.py
```

This compiles the AES entry points and dry-runs the CleanRL and skrl batch
launchers. It does not start long training jobs.

Toy smoke test:

```bash
cd toy
python train.py config=multigoal/flow_aes.yaml steps=120 warmup_steps=20 eval_every=50 plot_every=100000 device=cpu
```

CleanRL smoke test:

```bash
cd cleanrl
pip install -r requirements-aes.txt
python cleanrl/sac_continuous_action.py --env-id Pendulum-v1 --total-timesteps 200 --learning-starts 10 --batch-size 8 --use-aes --no-cuda --description runs_smoke/sac_aes
```

## Reproduction Entry Points

MuJoCo:

```bash
cd cleanrl
python aestd_runs.py --dry-run
python aestd_runs.py --algorithms sac ppo flow --envs Hopper-v4 HalfCheetah-v4 Walker2d-v4 Ant-v4 Humanoid-v4
python aestd_metrics.py --runs-dir runs_aes --output aes_metrics.csv
```

Isaac Gym:

```bash
cd skrl
PYTHON_PATH aestd_runs.py --dry-run
PYTHON_PATH aestd_train.py --algorithm flow --task Ant --use-nonstationary-env --drift-pattern abrupt --use-aes --output-dir runs_aes_isaac/flow/Ant/abrupt/aes/seed_1
```

See `AES_RELEASE.md` and `AES_EXPERIMENT_MATRIX.md` for the full experiment
matrix and implementation notes.

## License

This project is released under the MIT License. To preserve reproducibility,
parts of this repository are based on frozen versions of:

- `Toni-SM/skrl` at commit `631613a`
- `vwxyzjn/cleanrl` at commit `8cbca61`
- `VincentStimper/normalizing-flows` at commit `848277e`
- `rail-berkeley/softlearning` at commit `13cf187`

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{wang2026tracking,
    title={Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning},
    author={Wang, Tongxi and Xia, Zhuoyang and Chen, Xinran and Liu, Shan},
    booktitle={Proceedings of the 43rd International Conference on Machine Learning},
    year={2026}
}
```
