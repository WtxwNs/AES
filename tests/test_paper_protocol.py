import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'cleanrl/cleanrl'))
from aes import AESConfig, AESScheduler
from sql_continuous_action import soft_value, stein_direction
from nonstationary import make_task, maybe_wrap_nonstationary, set_global_step

spec = importlib.util.spec_from_file_location('metrics', ROOT / 'cleanrl/aestd_metrics.py')
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class PaperProtocolTests(unittest.TestCase):
    def test_scheduler_formula_and_resume(self):
        config = AESConfig(q=.5, beta=.5, kappa=.2, lambda_min=.01, lambda_max=10.)
        scheduler = AESScheduler(config)
        self.assertAlmostEqual(scheduler.update(torch.tensor([1.,3.])), .2*np.sqrt(2))
        self.assertAlmostEqual(scheduler.update(torch.tensor([3.,5.])), .2*np.sqrt(2.5))
        restored = AESScheduler(config)
        restored.load_state_dict(scheduler.state_dict())
        batch = torch.tensor([5.,7.])
        self.assertEqual(scheduler.update(batch), restored.update(batch))
        with self.assertRaises(ValueError):
            scheduler.update(torch.tensor([float('nan')]))
        with self.assertRaises(ValueError):
            AESScheduler(AESConfig(beta=1.))

    def test_sql_soft_integral_constant_energy_and_temperature(self):
        # Integral exp(c/alpha) over a box of volume 4 is exact for any particles.
        q = torch.full((2, 7), 3.)
        torch.testing.assert_close(soft_value(q, .5, np.log(4)), torch.full((2,), 3.+.5*np.log(4)))
        particles = torch.tensor([[[-1.],[1.]]])
        direction = stein_direction(particles, torch.zeros_like(particles))
        self.assertLess(direction[0,0,0], 0)
        self.assertGreater(direction[0,1,0], 0)

    def test_paper_metrics_hand_calculation_and_censoring(self):
        curve = [(0,10),(20,5),(40,10),(60,10),(80,10),(100,10)]
        self.assertEqual(metrics.auc(curve), 900.)
        rec, durations = metrics.recovery(curve, 100, [20,60])
        self.assertEqual(durations, [20.,0.])
        self.assertEqual(rec, .2)
        rec, durations = metrics.recovery([(0,10),(20,0),(40,5),(60,5),(100,5)],100,[20,60])
        self.assertIsNone(rec)
        self.assertEqual(durations, [None,0.])

    def test_table_uses_measured_paired_references_and_rejects_incomplete_curves(self):
        with tempfile.TemporaryDirectory() as tmp:
            for alg, variant, drift, value in [('sac','baseline','steady',10),('sac','aes','steady',12),('sac','aes','abrupt',9)]:
                folder=Path(tmp)/alg/'Hopper-v4'/drift/variant/'seed_1'
                folder.mkdir(parents=True)
                (folder/'evaluations.jsonl').write_text('\n'.join(json.dumps({'step':s,'return':value}) for s in [0,20,40,60,80,100]))
            rows=metrics.summarize(tmp,100,require_seeds=[1])
            row=next(r for r in rows if r['drift']=='abrupt')
            self.assertEqual(row['nauc'], .9)
            self.assertEqual(row['drop_ratio'], .25)
            (folder/'evaluations.jsonl').write_text(json.dumps({'step':20,'return':9}))
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                metrics.summarize(tmp,100)

    def test_toy_abrupt_jump_is_half_unit_and_boundary_is_stable(self):
        args=SimpleNamespace(use_nonstationary_env=True,drift_pattern='abrupt',total_timesteps=100,drift_fraction=.2)
        env=make_task('NonStationaryMultiGoal-v0',args)
        positions=[]
        for step in [0,20,40,60,80]:
            env.set_global_step(step)
            positions.append(env.goal_positions.copy())
        for a,b in zip(positions,positions[1:]):
            np.testing.assert_allclose(np.linalg.norm(a-b,axis=1),.5)
        env.set_global_step(100)
        np.testing.assert_array_equal(env.goal_positions,positions[-1])
        env.close()

    def test_missing_isaac_physics_hooks_fail_before_training(self):
        spec=importlib.util.spec_from_file_location('isaac',ROOT/'skrl/aes_isaac.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with self.assertRaisesRegex(RuntimeError,'validated adapter'):
            module.IsaacDriftWrapper(SimpleNamespace(),drift_pattern='abrupt')


if __name__=='__main__':
    unittest.main()
