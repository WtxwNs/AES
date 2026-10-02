"""Small CPU checks using the experiment dependencies, without training."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeTests(unittest.TestCase):
    def test_toy_seed_reproduces_reset_and_dynamics(self):
        module = load_module("multi_goal", "toy/toy_envs/multi_goal.py")
        for cls in (module.MultiGoal, module.NonStationaryMultiGoal):
            env = cls()
            env.dynamics.sigma = .1
            first, _ = env.reset(seed=7)
            transition = env.step(np.zeros(2))[0]
            np.random.normal(size=10)  # Other environments must not perturb this RNG.
            repeated, _ = env.reset(seed=7)
            np.testing.assert_array_equal(first, repeated)
            np.testing.assert_array_equal(transition, env.step(np.zeros(2))[0])
            different, _ = env.reset(seed=8)
            self.assertFalse(np.array_equal(first, different))
            env.close()

    def test_ppo_evaluation_returns_raw_rewards(self):
        import sys
        from types import SimpleNamespace
        import torch
        sys.path.insert(0, str(ROOT / "cleanrl/cleanrl"))
        ppo = load_module("ppo", "cleanrl/cleanrl/ppo_continuous_action.py")
        envs = ppo.gym.vector.SyncVectorEnv([ppo.make_env(
            "Pendulum-v1", 0, False, "test", .99, evaluation=True
        )])
        try:
            agent = SimpleNamespace(actor_mean=lambda obs: torch.zeros((len(obs), 1)))
            rms = SimpleNamespace(mean=np.zeros(3), var=np.ones(3))
            result = ppo.evaluate(envs, agent, "cpu", rms)
            env = ppo.gym.make("Pendulum-v1")
            env.reset(seed=0)
            expected = 0.0
            for _ in range(200):
                _, reward, _, _, _ = env.step(np.zeros(1))
                expected += reward
            env.close()
            self.assertAlmostEqual(result, expected, places=5)
        finally:
            envs.close()

    def test_value_plot_uses_cpu_policy_device(self):
        import tempfile
        import torch
        import matplotlib.pyplot as plt
        utils = load_module("toy_utils", "toy/modules/utils.py")
        class Critic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
            def get_v(self, obs):
                return obs.sum(dim=1, keepdim=True) * self.weight
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "value")
            utils.plot_value(Critic(), path)
            self.assertTrue(Path(path + ".png").is_file())
            plt.close("all")

    def test_metrics_auc(self):
        metrics = load_module("aestd_metrics", "cleanrl/aestd_metrics.py")
        self.assertAlmostEqual(metrics.auc([(0, 1), (2, 3), (4, 1)]), 2.0)
        self.assertTrue(np.isnan(metrics.auc([(0, 1)])))


if __name__ == "__main__":
    unittest.main()
