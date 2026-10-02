import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

# copy from https://github.com/rail-berkeley/softlearning/blob/master/softlearning/environments/gym/multi_goal.py
class MultiGoal(gym.Env):
    """
    Move a 2D point mass to one of the goal positions. Cost is the distance to
    the closest goal.

    State: position.
    Action: velocity.
    """
    def __init__(self,
                 goal_reward=10,
                 actuation_cost_coeff=30.0,
                 distance_cost_coeff=1.0,
                 init_sigma=0.1):
        super(MultiGoal, self).__init__()
        self.dynamics = PointDynamics(dim=2, sigma=0)
        self.init_mu = np.zeros(2, dtype=np.float32)
        self.init_sigma = init_sigma
        self.goal_positions = np.array(
            (
                (5, 0),
                (-5, 0),
                (0, 5),
                (0, -5)
            ),
            dtype=np.float32)
        self.goal_threshold = 1.0
        self.goal_reward = goal_reward
        self.action_cost_coeff = actuation_cost_coeff
        self.distance_cost_coeff = distance_cost_coeff
        self.xlim = (-7, 7)
        self.ylim = (-7, 7)
        self.vel_bound = 1.
        self.observation_space = spaces.Box(
            low=np.array((self.xlim[0], self.ylim[0])),
            high=np.array((self.xlim[1], self.ylim[1])),
            dtype=np.float32,
            shape=(2,),
            )
        self.action_space = spaces.Box(-self.vel_bound, self.vel_bound, shape=(2,), dtype=np.float32)
        self.reset()
        self.observation = None

        self._ax = None
        self._env_lines = []
        self.fixed_plots = None
        self.dynamic_plots = []
        self.timestep = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        unclipped_observation = (
            self.init_mu
            + self.init_sigma
            * self.np_random.normal(size=self.dynamics.s_dim))
        self.observation = np.clip(
            unclipped_observation,
            self.observation_space.low,
            self.observation_space.high).astype(np.float32)
        self.timestep = 0
        return self.observation, {'pos': self.observation}

    def get_current_obs(self):
        return np.copy(self.observation)

    def step(self, action):
        action = action.ravel()

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high).ravel()

        observation = self.dynamics.forward(self.observation, action, rng=self.np_random)
        observation = np.clip(
            observation,
            self.observation_space.low,
            self.observation_space.high).astype(np.float32)

        reward = self.compute_reward(observation, action)
        dist_to_goal = np.amin([
            np.linalg.norm(observation - goal_position)
            for goal_position in self.goal_positions
        ])
        done = dist_to_goal < self.goal_threshold
        if done:
            reward += self.goal_reward
        self.timestep += 1
        truncate = (self.timestep >= 1000)

        self.observation = np.copy(observation)

        return observation, reward, done, truncate, {'pos': observation}

    def _init_plot(self):
        fig_env = plt.figure(figsize=(7, 7))
        self._ax = fig_env.add_subplot(111)
        self._ax.axis('equal')

        self._env_lines = []
        self._ax.set_xlim((-7, 7))
        self._ax.set_ylim((-7, 7))

        self._ax.set_title('Multigoal Environment')
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')

        self._plot_position_cost(self._ax)

    def render_rollouts(self, paths=(), file_name="0"):
        """Render for rendering the past rollouts of the environment."""
        if self._ax is None:
            self._init_plot()

        # noinspection PyArgumentList
        [line.remove() for line in self._env_lines]
        self._env_lines = []

        for path in paths:
            # positions = np.stack(path['infos']['pos'])
            positions = np.stack(path)
            xx = positions[:, 0]
            yy = positions[:, 1]
            self._env_lines += self._ax.plot(xx, yy, 'b')

        # plt.draw()
        # plt.pause(0.01)
        plt.savefig(file_name+'.png')

    def render(self, mode='human', *args, **kwargs):
        """Render for rendering the current state of the environment."""
        pass

    def compute_reward(self, observation, action):
        # penalize the L2 norm of acceleration
        # noinspection PyTypeChecker
        action_cost = np.sum(action ** 2) * self.action_cost_coeff

        # penalize squared dist to goal
        cur_position = observation
        # noinspection PyTypeChecker
        goal_cost = self.distance_cost_coeff * np.amin([
            np.sum((cur_position - goal_position) ** 2)
            for goal_position in self.goal_positions
        ])

        # penalize staying with the log barriers
        costs = [action_cost, goal_cost]
        reward = -np.sum(costs)
        return reward

    def _plot_position_cost(self, ax):
        delta = 0.01
        x_min, x_max = tuple(1.1 * np.array(self.xlim))
        y_min, y_max = tuple(1.1 * np.array(self.ylim))
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )
        goal_costs = np.amin([
            (X - goal_x) ** 2 + (Y - goal_y) ** 2
            for goal_x, goal_y in self.goal_positions
        ], axis=0)
        costs = goal_costs

        contours = ax.contour(X, Y, costs, 20)
        ax.clabel(contours, inline=1, fontsize=10, fmt='%.0f')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        goal = ax.plot(self.goal_positions[:, 0],
                       self.goal_positions[:, 1], 'ro')
        return [contours, goal]


class PointDynamics(object):
    """
    State: position.
    Action: velocity.
    """
    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma
        self.s_dim = dim
        self.a_dim = dim

    def forward(self, state, action, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        mu_next = state + action
        state_next = mu_next + self.sigma * \
            rng.normal(size=self.s_dim)
        return state_next


class NonStationaryMultiGoal(MultiGoal):
    """MultiGoal with goal drift protocols from the Tracking Drift appendix."""

    def __init__(
        self,
        drift_pattern="steady",
        total_steps=4000,
        drift_fraction=0.2,
        abrupt_distance=0.5,
        periodic_amplitude=0.25,
        **kwargs,
    ):
        self.drift_pattern = drift_pattern
        self.total_steps = max(int(total_steps), 1)
        self.drift_fraction = float(drift_fraction)
        self.abrupt_distance = float(abrupt_distance)
        self.periodic_amplitude = float(periodic_amplitude)
        self.global_step = 0
        self._drift_direction = np.array((1.0, 0.0), dtype=np.float32)
        super().__init__(**kwargs)
        self.base_goal_positions = self.goal_positions.copy()
        self._apply_drift()

    def set_global_step(self, step):
        self.global_step = int(step)
        self._apply_drift()

    def reset(self, seed=None, options=None):
        self._apply_drift()
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self._apply_drift()
        return super().step(action)

    def _apply_drift(self):
        if not hasattr(self, "base_goal_positions"):
            return
        progress = np.clip(self.global_step / self.total_steps, 0.0, 1.0)
        shift = self._goal_shift(progress)
        self.goal_positions = self.base_goal_positions + shift.astype(np.float32)

    def _goal_shift(self, progress):
        if self.drift_pattern == "steady":
            return np.zeros(2, dtype=np.float32)

        segment = max(self.drift_fraction, 1e-6)
        local = (progress % segment) / segment

        if self.drift_pattern == "abrupt":
            jumps = int(progress / segment)
            if jumps == 0:
                return np.zeros(2, dtype=np.float32)
            sign = 1.0 if jumps % 2 else -1.0
            return sign * self.abrupt_distance * self._drift_direction

        if self.drift_pattern == "linear":
            if progress < segment:
                return np.zeros(2, dtype=np.float32)
            return self.abrupt_distance * local * self._drift_direction

        if self.drift_pattern == "periodic":
            phase = 2.0 * np.pi * progress / segment
            return self.periodic_amplitude * np.array((np.sin(phase), np.cos(phase)), dtype=np.float32)

        if self.drift_pattern == "mixed":
            jumps = int(progress / segment)
            if jumps == 0:
                return np.zeros(2, dtype=np.float32)
            abrupt = (0.5 if jumps % 2 == 0 else -0.5) * self.abrupt_distance * self._drift_direction
            linear = 0.5 * self.abrupt_distance * local * np.array((0.0, 1.0), dtype=np.float32)
            return abrupt + linear

        raise ValueError(f"Unknown drift_pattern: {self.drift_pattern}")
