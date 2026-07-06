from gymnasium.envs.registration import register
from .multi_goal import MultiGoal, NonStationaryMultiGoal

register(
    id='MultiGoal-v0',
    entry_point='toy_envs:MultiGoal',
)

register(
    id='NonStationaryMultiGoal-v0',
    entry_point='toy_envs:NonStationaryMultiGoal',
)
