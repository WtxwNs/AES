"""Recompute paper metrics from complete, measured evaluation curves.

Equations (32)-(35) and Appendix C.9. No missing run is filled from a paper
number. Recovery is right-censored at the next shift (or end of training).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_scalar_series(run_dir, tag="Test/return"):
    path = Path(run_dir) / "evaluations.jsonl"
    if path.exists():
        points = [(r["step"], r["return"]) for r in
                  (json.loads(line) for line in path.read_text().splitlines() if line.strip())]
    else:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        points = []
        for event_file in sorted(Path(run_dir).rglob("events.out.tfevents.*")):
            acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            acc.Reload()
            if tag in acc.Tags().get("scalars", []):
                points.extend((event.step, event.value) for event in acc.Scalars(tag))
    # Identical duplicates are harmless; conflicting restarts must be resolved explicitly.
    unique = {}
    for step, value in points:
        if step in unique and not np.isclose(unique[step], value, rtol=1e-6, atol=1e-7):
            raise ValueError(f"Conflicting evaluations at step {step} in {run_dir}")
        if step < 0 or not np.isfinite(value):
            raise ValueError(f"Invalid evaluation point in {run_dir}")
        unique[step] = value
    return sorted(unique.items())


def auc(points):
    """Unnormalized integral, in return * environment steps (paper Eq. 32)."""
    if len(points) < 2:
        return np.nan
    x, y = np.asarray(points, dtype=float).T
    if np.any(np.diff(x) <= 0) or not np.isfinite(y).all():
        raise ValueError("AUC requires finite returns and strictly increasing steps")
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2))


def recovery(points, horizon, change_points):
    """First observed recovery to last pre-change return; no interpolation.

    A missing recovery is None, never zero or a fabricated finite time.
    Report the sum / T only if every change has a recovery before the next one.
    """
    x, y = np.asarray(points, dtype=float).T
    durations = []
    for i, cp in enumerate(change_points):
        stop = change_points[i + 1] if i + 1 < len(change_points) else horizon + 1
        before = np.flatnonzero(x < cp)
        if not len(before):
            raise ValueError("Recovery requires a pre-change observation")
        after = np.flatnonzero((x >= cp) & (x < stop) & (y >= y[before[-1]]))
        durations.append(None if not len(after) else float(x[after[0]] - cp))
    value = None if any(v is None for v in durations) else sum(durations) / horizon
    return value, durations


def summarize(root, horizon, tag="Test/return", drift_fraction=.2, require_seeds=None):
    rows = []
    for directory in sorted(Path(root).glob("*/*/*/*/seed_*")):
        algorithm, task, drift, variant, seed = directory.relative_to(root).parts
        points = read_scalar_series(directory, tag)
        if not points or points[0][0] != 0 or points[-1][0] != horizon:
            raise ValueError(f"Incomplete [0, {horizon}] evaluation curve: {directory}")
        metadata = directory / "run.json"
        fraction = drift_fraction
        if metadata.exists():
            settings = json.loads(metadata.read_text())["args"]
            if settings["total_timesteps"] != horizon:
                raise ValueError(f"Mixed training horizons: {directory}")
            fraction = settings.get("drift_fraction", fraction)
        cps = [int(round(horizon * fraction * j)) for j in range(1, int(np.ceil(1 / fraction)))
               if fraction * j < 1 - 1e-10]
        rec, durations = recovery(points, horizon, cps) if drift == "abrupt" else (None, [])
        family = "Toy" if "MultiGoal" in task else "MuJoCo"
        rows.append(dict(algorithm=algorithm, env_id=task, family=family, drift=drift,
                         variant=variant, seed=int(seed.removeprefix("seed_")),
                         num_eval_points=len(points), auc=auc(points),
                         final_return=points[-1][1], recovery=rec,
                         censored_changes=sum(d is None for d in durations),
                         recovery_steps=json.dumps(durations)))
    if not rows:
        raise ValueError(f"No runs found in {root}")
    if require_seeds:
        groups = {}
        for row in rows:
            key = (row["algorithm"], row["env_id"], row["drift"], row["variant"])
            groups.setdefault(key, set()).add(row["seed"])
        for key, seeds in groups.items():
            if seeds != set(require_seeds):
                raise ValueError(f"Incomplete seed set {key}: {sorted(seeds)}")
    steady = {(r["algorithm"], r["env_id"], r["variant"], r["seed"]): r["auc"]
              for r in rows if r["drift"] == "steady"}
    # Paper specifies family Steady SAC reference, then equal-weight task averages.
    reference = {}
    for family in {r["family"] for r in rows}:
        task_means = []
        for task in {r["env_id"] for r in rows if r["family"] == family}:
            values = [r["auc"] for r in rows if r["env_id"] == task and
                      r["algorithm"] == "sac" and r["variant"] == "baseline" and r["drift"] == "steady"]
            if not values:
                raise ValueError(f"Missing Steady SAC normalization reference for {task}")
            task_means.append(np.mean(values))
        reference[family] = float(np.mean(task_means))
    for row in rows:
        key = (row["algorithm"], row["env_id"], row["variant"], row["seed"])
        if key not in steady:
            raise ValueError(f"Missing paired Steady run: {key}")
        denom = reference[row["family"]]
        if abs(denom) < 1e-12 or abs(steady[key]) < 1e-12:
            raise ValueError("Zero AUC denominator; the paper ratios are undefined")
        row["nauc"] = row["auc"] / denom
        row["drop_ratio"] = 1 - row["auc"] / steady[key]
        row["reference_auc"] = denom
        row["ratio_direction_valid"] = denom > 0 and steady[key] > 0
    return rows


def aggregate(rows):
    """Equal-weight tasks, with sample SD across paired seed aggregates."""
    groups = {}
    for r in rows:
        key = (r["family"], r["algorithm"], r["drift"], r["variant"])
        groups.setdefault(key, []).append(r)
    result = []
    for key, group in sorted(groups.items()):
        row = dict(zip(("family", "algorithm", "drift", "variant"), key))
        row["tasks"] = len({r["env_id"] for r in group})
        row["seeds"] = len({r["seed"] for r in group})
        for metric in ("nauc", "drop_ratio", "recovery"):
            values = []
            for seed in sorted({r["seed"] for r in group}):
                part = [r[metric] for r in group if r["seed"] == seed]
                if len(part) == row["tasks"] and all(v is not None for v in part):
                    values.append(float(np.mean(part)))
            complete = len(values) == row["seeds"]
            row[metric + "_mean"] = float(np.mean(values)) if values and complete else None
            row[metric + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 and complete else None
        row["censored_changes"] = sum(r["censored_changes"] for r in group)
        result.append(row)
    return result


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs_aes")
    parser.add_argument("--tag", default="Test/return")
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--drift-fraction", type=float, default=.2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--output", default="aes_metrics.csv")
    parser.add_argument("--allow-partial-matrix", action="store_true", help="Label output as partial; do not claim a paper table")
    args = parser.parse_args()
    rows = summarize(args.runs_dir, args.horizon, args.tag, args.drift_fraction, args.seeds)
    if not args.allow_partial_matrix:
        expected_tasks = {"MuJoCo": {"Hopper-v4", "HalfCheetah-v4", "Walker2d-v4", "Ant-v4", "Humanoid-v4"},
                          "Toy": {"NonStationaryMultiGoal-v0"}}
        present = {(r["family"],r["env_id"],r["algorithm"],r["drift"],r["variant"],r["seed"]) for r in rows}
        import itertools
        for family in {r["family"] for r in rows}:
            expected = {(family,*cell) for cell in itertools.product(expected_tasks[family],
                        ["sac","ppo","sql","flow"], ["steady","abrupt","linear","periodic","mixed"],
                        ["baseline","aes"], args.seeds)}
            missing = expected - present
            if missing:
                raise ValueError(f"Missing {len(missing)} paper cells in {family}; use --allow-partial-matrix for a development report")
    else:
        print("PARTIAL MATRIX: development summary, not a complete paper table")
    write_csv(args.output, rows)
    write_csv(Path(args.output).with_suffix(".summary.csv"), aggregate(rows))
    if any(not row["ratio_direction_valid"] for row in rows):
        print("CAUTION: non-positive reference AUC; paper ratios do not preserve higher/lower-is-better ordering. Raw AUC is retained.")
    print(f"Wrote {len(rows)} measured runs; no paper values were used as measurements.")


if __name__ == "__main__":
    main()
