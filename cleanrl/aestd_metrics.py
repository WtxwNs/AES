"""Aggregate TensorBoard evaluation logs for AES experiments."""

import argparse
import csv
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_scalar_series(run_dir, tag):
    points = []
    for event_file in Path(run_dir).rglob("events.out.tfevents.*"):
        acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        acc.Reload()
        if tag not in acc.Tags().get("scalars", []):
            continue
        points.extend((event.step, event.value) for event in acc.Scalars(tag))
    points.sort()
    return points


def auc(points):
    if len(points) < 2:
        return np.nan
    x = np.array([p[0] for p in points], dtype=np.float64)
    y = np.array([p[1] for p in points], dtype=np.float64)
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0) / max(x[-1] - x[0], 1.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs_aes")
    parser.add_argument("--tag", default="Test/return")
    parser.add_argument("--output", default="aes_metrics.csv")
    args = parser.parse_args()

    rows = []
    root = Path(args.runs_dir)
    for run_dir in root.glob("*/*/*/*/seed_*"):
        rel = run_dir.relative_to(root).parts
        if len(rel) != 5:
            continue
        algorithm, env_id, drift, variant, seed = rel
        points = read_scalar_series(run_dir, args.tag)
        rows.append(
            {
                "algorithm": algorithm,
                "env_id": env_id,
                "drift": drift,
                "variant": variant,
                "seed": seed.replace("seed_", ""),
                "num_eval_points": len(points),
                "auc": auc(points),
                "final_return": np.nan if not points else points[-1][1],
            }
        )

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["algorithm", "env_id", "drift", "variant", "seed", "num_eval_points", "auc", "final_return"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
