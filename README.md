# Tracking Drift: Adaptive Entropy Scheduling (AES)

Code for *Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary
Reinforcement Learning* ([paper](https://arxiv.org/abs/2601.19624)).

AES uses the 0.9 quantile of absolute TD residuals, EMA smoothing, and the
clipped square root of their running mean to schedule the entropy coefficient.
The paired baseline and AES runs keep the carrier's other settings identical.

**Reproduction status:** Toy and MuJoCo have executable SAC, PPO, SQL and
flow-based carrier paths. SQL is a newly reconstructed SVGD implementation.
Isaac experiments and historical numerical replication remain incomplete.
Read [REPRODUCTION.md](REPRODUCTION.md) for the paper-to-code map and the
precise remaining inputs. A smoke test is not evidence of the paper's scores.

## Install and verify

Portable validated stack: Python 3.12; CPU or a compatible PyTorch GPU build.
The original paper reports Python 3.9 / PyTorch 1.13 / CUDA 11.7; the portable
stack is a reconstruction, not an archived original environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r cleanrl/requirements-aes.txt
python scripts/check_release.py
python -m unittest discover -s tests -v
```

For CPU-only machines, install `torch==2.5.1` from the PyTorch CPU wheel index
before the requirements. For GPUs needing a newer PyTorch/CUDA build, record
that environment separately and rerun the checks; it is not the validated stack.

## Run a real training smoke test

From the repository root:

```bash
python cleanrl/cleanrl/sac_continuous_action.py \
  --env-id Pendulum-v1 --total-timesteps 40 --learning-starts 8 \
  --batch-size 8 --buffer-size 128 --eval-frequency 20 \
  --use-aes --no-cuda --description runs_smoke/sac
```

`Pendulum-v1` tests the training interface only; it is not a paper task and has
no injected physics drift. Substitute `Hopper-v4` and add
`--use-nonstationary-env --drift-pattern abrupt` to exercise MuJoCo drift.
For SQL use `sql_continuous_action.py`; for the flow carrier use
`flow_continuous_action.py`. PPO uses `ppo_continuous_action.py` with
`--num-steps 20 --num-minibatches 2 --update-epochs 2` instead of replay options.

## Enumerate or execute the experiment matrix

The full paper grid has **2,400 runs**: 12 tasks × 4 carriers × 5 drift
patterns × 2 variants × 5 seeds. Of these, 1,200 Toy/MuJoCo cells have runnable
entry points. The manifest retains the 1,200 blocked Isaac cells explicitly.

```bash
python scripts/paper_matrix.py --horizon 1000000 --manifest paper_matrix.json
python scripts/paper_matrix.py --family mujoco --horizon 1000000 \
  --output-dir runs_paper --manifest mujoco_matrix.json --execute
python scripts/paper_matrix.py --family toy --horizon 1000000 \
  --output-dir runs_paper --manifest toy_matrix.json --execute
```

Here `1000000` is an explicit reconstruction budget, not a recovered per-task
paper setting. The launcher uses complete 1,000-step PPO rollouts with 10
minibatches; it records this choice. Supply the original settings when available.
Run output directories cannot be reused accidentally.

## Recompute results

Every portable carrier writes `run.json`, `evaluations.jsonl`, TensorBoard
scalars, and a state-dict `final.pt`. Evaluation uses 10 deterministic episodes,
raw undiscounted rewards, and the current drift setting, at steps 0, every
10,000 interactions, and the training endpoint. PPO evaluates at the exact
interaction grid inside rollouts, before a policy update if the two coincide.

```bash
python cleanrl/aestd_metrics.py --runs-dir runs_paper/mujoco \
  --horizon 1000000 --output results/mujoco.csv
```

The script recomputes the integral AUC, paper nAUC, paired DropRatio, and
abrupt-change recovery. It requires complete curves, five seeds and the full
selected-family matrix. `--allow-partial-matrix --seeds 1` creates an explicitly
partial development report. Censored recovery stays missing. Negative reference
AUC is flagged because the paper ratios then lose their stated direction.
No paper-table value is inserted into a measured result.

## Layout and provenance

- `configs/paper_protocol.json`: paper settings and unresolved inputs.
- `cleanrl/cleanrl/`: portable carriers, scheduler, drift and run records.
- `scripts/paper_matrix.py`: complete matrix enumeration and execution.
- `cleanrl/aestd_metrics.py`: strict result reconstruction.
- `toy/`: original flow experiment path; portable matrix uses its same environment.
- `skrl/`: legacy Isaac integrations; non-steady runs now require actual physics setters.
- `external_sql_aes_adapter.py`: retained external integration helper.
- [VALIDATION.md](VALIDATION.md): commands actually run during this revision.

Vendored upstream code retains its licenses. Historical upstream revisions are
`Toni-SM/skrl@631613a`, `vwxyzjn/cleanrl@8cbca61`,
`VincentStimper/normalizing-flows@848277e`, and
`rail-berkeley/softlearning@13cf187`.

```bibtex
@misc{wang2026trackingdriftvariationawareentropy,
  title={Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning},
  author={Tongxi Wang and Zhuoyang Xia and Xinran Chen and Shan Liu},
  year={2026}, eprint={2601.19624}, archivePrefix={arXiv}, primaryClass={cs.LG}
}
```
