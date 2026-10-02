import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

COMPILE_TARGETS = [
    "cleanrl/aestd_runs.py",
    "cleanrl/aestd_metrics.py",
    "cleanrl/cleanrl/aes.py",
    "cleanrl/cleanrl/nonstationary.py",
    "cleanrl/cleanrl/sac_continuous_action.py",
    "cleanrl/cleanrl/ppo_continuous_action.py",
    "cleanrl/cleanrl/flow_continuous_action.py",
    "skrl/aes_isaac.py",
    "skrl/aestd_train.py",
    "skrl/aestd_runs.py",
    "skrl/skrl/agents/torch/aes.py",
    "skrl/skrl/agents/torch/sac/sac.py",
    "skrl/skrl/agents/torch/flow/flow.py",
    "toy/modules/aes.py",
    "toy/agents/flow.py",
    "toy/modules/train_loop.py",
    "toy/toy_envs/multi_goal.py",
    "toy/train.py",
    "external_sql_aes_adapter.py",
]

REQUIRED_FILES = [
    "AES_RELEASE.md",
    "AES_EXPERIMENT_MATRIX.md",
    "cleanrl/requirements-aes.txt",
    "toy/REPRODUCE_TRACKING_DRIFT.md",
]


def run(command, cwd=ROOT):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def assert_exists():
    missing = [path for path in REQUIRED_FILES + COMPILE_TARGETS if not (ROOT / path).exists()]
    if missing:
        raise SystemExit("Missing release files:\n" + "\n".join(missing))


def compile_targets():
    run([sys.executable, "-m", "compileall", "-q", *COMPILE_TARGETS])


def dry_run_launchers():
    run([
        sys.executable,
        "cleanrl/aestd_runs.py",
        "--dry-run",
        "--algorithms",
        "sac",
        "--envs",
        "Pendulum-v1",
        "--drifts",
        "abrupt",
        "--seeds",
        "1",
        "--total-timesteps",
        "1000",
        "--no-cuda",
    ])
    run([
        sys.executable,
        "skrl/aestd_runs.py",
        "--dry-run",
        "--algorithms",
        "sac",
        "--tasks",
        "Ant",
        "--drifts",
        "abrupt",
        "--seeds",
        "1",
        "--timesteps",
        "1000",
    ])


def assert_no_generated_python_cache():
    caches = [str(path.relative_to(ROOT)) for path in ROOT.rglob("__pycache__")]
    if caches:
        raise SystemExit("Generated __pycache__ directories found:\n" + "\n".join(caches[:20]))


def cleanup_python_cache():
    for path in sorted(ROOT.rglob("__pycache__"), reverse=True):
        for child in path.iterdir():
            child.unlink()
        path.rmdir()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cache-check", action="store_true")
    args = parser.parse_args()

    assert_exists()
    compile_targets()
    dry_run_launchers()
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_release_regressions.py"])
    cleanup_python_cache()
    if not args.skip_cache_check:
        assert_no_generated_python_cache()
    print("Release check passed.")


if __name__ == "__main__":
    main()
