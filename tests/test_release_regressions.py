"""Lightweight regression tests; no training or simulator installation needed."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LauncherTests(unittest.TestCase):
    def test_ppo_accepts_every_batch_launcher_option(self):
        launcher = load_module("cleanrl_launcher", "cleanrl/aestd_runs.py")
        args = SimpleNamespace(output_dir="runs", total_timesteps=1000,
                               eval_frequency=100, aes_q=.9, aes_beta=.95,
                               aes_kappa=1., cuda=False, num_envs=1, num_steps=32)
        command = launcher.build_command(args, "ppo", "Pendulum-v1", "steady", 7, True)
        tree = ast.parse((ROOT / "cleanrl/cleanrl/ppo_continuous_action.py").read_text())
        fields = {node.target.id for node in next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == "Args").body
                  if isinstance(node, ast.AnnAssign)}
        for option in (part for part in command if part.startswith("--")):
            field = option[2:].removeprefix("no-").replace("-", "_")
            self.assertIn(field, fields)

    def test_isaac_command_passes_seed(self):
        launcher = load_module("isaac_launcher", "skrl/aestd_runs.py")
        args = SimpleNamespace(output_dir="runs", timesteps=10, num_envs=1, batch_size=2)
        command = launcher.command(args, "sac", "Ant", "steady", 7, True)
        self.assertEqual(command[command.index("--seed") + 1], "7")

    def test_isaac_trainers_consume_config_seed(self):
        for filename in ("skrl/trainer_sac.py", "skrl/trainer_flow.py"):
            tree = ast.parse((ROOT / filename).read_text())
            train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_train")
            seed_call = next(n for n in ast.walk(train) if isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Name) and n.func.id == "set_seed")
            self.assertEqual(ast.unparse(seed_call.args[0]), "cfg.get('seed')")

    def test_ppo_evaluates_when_rollout_crosses_interval(self):
        tree = ast.parse((ROOT / "cleanrl/cleanrl/ppo_continuous_action.py").read_text())
        condition = next(n.test for n in ast.walk(tree) if isinstance(n, ast.If)
                         and "next_eval_step" in ast.unparse(n.test))
        code = compile(ast.Expression(condition), "<evaluation condition>", "eval")
        args = SimpleNamespace(eval_frequency=10000)
        next_eval_step = args.eval_frequency
        evaluations = []
        for global_step in range(2048, 1000001, 2048):
            if eval(code, {}, dict(args=args, global_step=global_step, next_eval_step=next_eval_step)):
                evaluations.append(global_step)
                next_eval_step = (global_step // args.eval_frequency + 1) * args.eval_frequency
        self.assertEqual(evaluations[0], 10240)
        self.assertEqual(len(evaluations), 99)


if __name__ == "__main__":
    unittest.main()
