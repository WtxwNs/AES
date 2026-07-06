"""Batch launcher for AES CleanRL experiments."""

import argparse
import itertools
import os
import subprocess
import sys


ALGORITHMS = {
    "sac": "cleanrl/sac_continuous_action.py",
    "ppo": "cleanrl/ppo_continuous_action.py",
    "flow": "cleanrl/flow_continuous_action.py",
}

MUJOCO_ENVS = ["Hopper-v4", "HalfCheetah-v4", "Walker2d-v4", "Ant-v4", "Humanoid-v4"]
DRIFTS = ["steady", "abrupt", "linear", "periodic", "mixed"]
SEEDS = [1, 2, 3, 4, 5]


def build_command(args, algorithm, env_id, drift, seed, aes):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), ALGORITHMS[algorithm])
    tag = "aes" if aes else "baseline"
    out_dir = os.path.join(args.output_dir, algorithm, env_id, drift, tag, f"seed_{seed}")
    command = [
        sys.executable,
        script,
        "--env-id",
        env_id,
        "--seed",
        str(seed),
        "--total-timesteps",
        str(args.total_timesteps),
        "--use-nonstationary-env",
        "--drift-pattern",
        drift,
        "--description",
        out_dir,
        "--eval-frequency",
        str(args.eval_frequency),
    ]
    if aes:
        command += [
            "--use-aes",
            "--aes-q",
            str(args.aes_q),
            "--aes-beta",
            str(args.aes_beta),
            "--aes-kappa",
            str(args.aes_kappa),
        ]
    if args.cuda is False:
        command += ["--no-cuda"]
    if algorithm == "ppo":
        command += ["--num-envs", str(args.num_envs), "--num-steps", str(args.num_steps)]
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", nargs="+", default=["sac", "ppo", "flow"], choices=ALGORITHMS)
    parser.add_argument("--envs", nargs="+", default=MUJOCO_ENVS)
    parser.add_argument("--drifts", nargs="+", default=DRIFTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--output-dir", default="runs_aes")
    parser.add_argument("--eval-frequency", type=int, default=10_000)
    parser.add_argument("--aes-q", type=float, default=0.9)
    parser.add_argument("--aes-beta", type=float, default=0.95)
    parser.add_argument("--aes-kappa", type=float, default=1.0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for algorithm, env_id, drift, seed, aes in itertools.product(
        args.algorithms, args.envs, args.drifts, args.seeds, [False, True]
    ):
        command = build_command(args, algorithm, env_id, drift, seed, aes)
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
