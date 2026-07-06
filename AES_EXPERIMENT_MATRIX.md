# AES experiment matrix

This matrix maps the paper experiments to repository entry points.

## Toy

- Carrier: flow-based policy
- Entry points:
  - Baseline: `toy/conf/config/multigoal/flow_nonstationary.yaml`
  - AES: `toy/conf/config/multigoal/flow_aes.yaml`
- Drifts: `steady`, `abrupt`, `linear`, `periodic`, `mixed`
- Channel: 2D goal relocation

## MuJoCo

- Carriers: SAC, PPO, flow-based policy
- Entry points:
  - `cleanrl/cleanrl/sac_continuous_action.py`
  - `cleanrl/cleanrl/ppo_continuous_action.py`
  - `cleanrl/cleanrl/flow_continuous_action.py`
  - Batch: `cleanrl/aestd_runs.py`
- Tasks: `Hopper-v4`, `HalfCheetah-v4`, `Walker2d-v4`, `Ant-v4`, `Humanoid-v4`
- Drifts: `steady`, `abrupt`, `linear`, `periodic`, `mixed`
- Channel: body mass and joint friction scaling

## Isaac Gym

- Carriers: SAC, flow-based policy
- Entry points:
  - Single run: `skrl/aestd_train.py`
  - Batch: `skrl/aestd_runs.py`
- Tasks: `AllegroHand`, `Ant`, `Anymal`, `FrankaCabinet`, `Humanoid`, `Ingenuity`
- Drifts: `steady`, `abrupt`, `linear`, `periodic`, `mixed`
- Channel: gravity and task-exposed joint damping hooks

## SQL

- Local status: external adapter template only
- Entry point: `external_sql_aes_adapter.py`
- Reason: this repository does not vendor a SQL implementation.
- Required insertion point: replace the static SQL temperature in the soft
  Bellman target / soft value computation with the AES scheduled value.

## Seeds

Use `{1, 2, 3, 4, 5}` for reported runs.

## AES hyperparameters

- `q = 0.9`
- `beta = 0.95`
- `kappa = 1.0`
- Off-policy carrier range: `[1e-4, 1.0]`
- PPO range: `[1e-4, 0.1]`
