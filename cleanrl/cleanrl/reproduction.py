"""Small, shared run records for the paper's evaluation protocol."""
import json
import platform
from pathlib import Path

import numpy as np
import torch


def write_run_config(directory, args):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    if any((path / name).exists() for name in ("run.json", "evaluations.jsonl", "final.pt")) or any(path.glob("events.out.tfevents.*")):
        raise FileExistsError(f"Run already exists: {path}; use a new output directory")
    data = {"schema_version": 1, "args": vars(args), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__,
            "evaluation": "deterministic, 10 episodes, undiscounted raw return"}
    (path / "run.json").write_text(json.dumps(data, indent=2) + "\n")


def log_evaluation(writer, directory, step, result):
    result = float(result)
    if not np.isfinite(result):
        raise ValueError("Non-finite evaluation return")
    writer.add_scalar("Test/return", result, step)
    writer.flush()
    with (Path(directory) / "evaluations.jsonl").open("a") as f:
        f.write(json.dumps({"step": int(step), "return": result}) + "\n")


def save_training_state(directory, **state):
    """Save weights/config/scheduler without pickling live model objects.

    This is a final training snapshot, not an exact restart of the simulator
    and replay buffer. Do not describe it as an exact-resume checkpoint.
    """
    torch.save(state, Path(directory) / "final.pt")
