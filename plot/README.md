# Tracking Drift Plot Utilities

This directory contains plotting utilities for the ICML 2026 paper:

**Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning**

The scripts expect TensorBoard event files produced by the experiment launchers in
`cleanrl/`, `skrl/`, and `toy/`. They are intentionally lightweight: the release
ships the plotting entry points and directory conventions, while generated logs
and rendered figures are ignored by git.

## Install

From the repository root:

```bash
pip install tbparse seaborn pandas matplotlib
```

For the full experiment environments, install the dependencies listed in the
corresponding backend directories:

- `cleanrl/requirements-aes.txt` for MuJoCo/Gymnasium experiments.
- `toy/requirements.txt` for the non-stationary toy environments.
- `skrl/` environment dependencies for Isaac Gym / Isaac Lab experiments.

## Expected Layout

Place experiment outputs under a local results directory such as:

```text
smoothed/
  Hopper-v4/
    sac_aes/
      1/
        events.out.tfevents...
      2/
      3/
  Walker2d-v4/
  Ant-v4/
```

The exact algorithm labels should match the names emitted by the launch scripts
or the post-processing step used for the paper figures.

## Plotting

Run the figure scripts from this directory:

```bash
python plot_fig_3.py
```

The existing figure scripts preserve the original plotting interfaces and are
intended to be used with the experiment matrix in `../AES_EXPERIMENT_MATRIX.md`.
Generated figure folders are local artifacts and should not be committed.

## Notes

- `cleanrl/aestd_metrics.py` provides scalar extraction and aggregation helpers
  for TensorBoard logs.
- `AES_EXPERIMENT_MATRIX.md` documents the paper-level experiment coverage,
  including dynamic variation schedules and AES ablations.
- `AES_RELEASE.md` summarizes the implemented release surface and validation
  checks.

## Citation

```bibtex
@inproceedings{wang2026trackingdrift,
    title={Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning},
    author={Wang, Tongxi and Xia, Zhuoyang and Chen, Xinran and Liu, Shan},
    booktitle={Proceedings of the 43rd International Conference on Machine Learning},
    year={2026}
}
```
