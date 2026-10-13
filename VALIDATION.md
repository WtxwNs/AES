# Executed validation

Checked on 2026-10-13 with Python 3.12, PyTorch 2.5.1+cpu,
NumPy 1.26.4, Gymnasium 0.29.1, Stable-Baselines3 2.3.2 and MuJoCo
3.1.6. No CUDA device or Isaac simulator was used.

## Automated checks

```bash
python -m unittest discover -s tests -v
python scripts/check_release.py
python scripts/paper_matrix.py --horizon 1000000 --manifest paper_matrix.json
```

- All **14 tests passed**. These cover the scheduler equation/state, SQL soft
  integral and particle update, PPO evaluation rewards, exact evaluation timing,
  Toy seeds/drift boundaries, AUC/reference pairing/recovery censoring, and
  failure before training when Isaac physics hooks are absent.
- Release compilation and launcher checks passed.
- The manifest contains **2,400 cells**, with **1,200 runnable Toy/MuJoCo cells**
  and **1,200 blocked Isaac cells**. Enumeration does not mean those runs executed.

## Actual training and reporting

The measured arguments and evaluation points are retained in
[`validation/cpu_smoke_results.json`](validation/cpu_smoke_results.json).

Ten short CPU jobs completed with seed 1 and 40 environment interactions each:

| Environment | Carrier | Variants | Drift |
|---|---|---|---|
| Pendulum-v1 | SAC, PPO, SQL, flow | baseline and AES (8 runs) | steady |
| Hopper-v4 | SAC | AES | abrupt |
| NonStationaryMultiGoal-v0 | PPO | AES | abrupt |

Every run wrote evaluations at **0, 20, 40**, run metadata, TensorBoard events
and a `final.pt` state-dict checkpoint. Evaluations used 10 episodes. Pendulum
is an interface fixture, not a paper task.

Common flags were `--total-timesteps 40 --eval-frequency 20 --seed 1 --no-cuda`.
Off-policy carriers additionally used `--learning-starts 8 --batch-size 8
--buffer-size 128`. SQL used `--kernel-particles 4 --value-particles 8`.
PPO used `--num-steps 20 --num-minibatches 2 --update-epochs 2`.
The two drift jobs used `--use-nonstationary-env --drift-pattern abrupt`.
Each run had a separate `--description` output directory; AES jobs added
`--use-aes`. These settings exercise code paths, not learning performance.

The eight paired Pendulum logs were then passed to the reporter using:

```bash
python cleanrl/aestd_metrics.py --runs-dir PATH_TO_PAIRED_PENDULUM_RUNS \
  --horizon 40 --seeds 1 --allow-partial-matrix --output results/smoke.csv
```

It produced eight measured rows plus their aggregate CSV, labeled the matrix
partial, and flagged the negative AUC denominator. No published score was used
as an observation. The unpaired Hopper/Toy smoke jobs were excluded from that
paired report, rather than supplied invented Steady references.

## Not established by these checks

No full training matrix, five-seed statistical result, Isaac performance,
original SQL equivalence or paper-table agreement has been established.
The complete remaining requirements are in [REPRODUCTION.md](REPRODUCTION.md).
