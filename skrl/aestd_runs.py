import argparse
import itertools
import os
import subprocess
import sys


TASKS = ["AllegroHand", "Ant", "Anymal", "FrankaCabinet", "Humanoid", "Ingenuity"]
DRIFTS = ["steady", "abrupt", "linear", "periodic", "mixed"]
ALGORITHMS = ["sac", "flow"]
SEEDS = [1, 2, 3, 4, 5]


def command(args, algorithm, task, drift, seed, aes):
    variant = "aes" if aes else "baseline"
    out_dir = os.path.join(args.output_dir, algorithm, task, drift, variant, f"seed_{seed}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        os.path.join(script_dir, "aestd_train.py"),
        "--algorithm",
        algorithm,
        "--task",
        task,
        "--timesteps",
        str(args.timesteps),
        "--num-envs",
        str(args.num_envs),
        "--batch-size",
        str(args.batch_size),
        "--use-nonstationary-env",
        "--drift-pattern",
        drift,
        "--output-dir",
        out_dir,
    ]
    if aes:
        cmd += ["--use-aes"]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS, choices=ALGORITHMS)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--drifts", nargs="+", default=DRIFTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default="runs_aes_isaac")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for algorithm, task, drift, seed, aes in itertools.product(
        args.algorithms, args.tasks, args.drifts, args.seeds, [False, True]
    ):
        cmd = command(args, algorithm, task, drift, seed, aes)
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
