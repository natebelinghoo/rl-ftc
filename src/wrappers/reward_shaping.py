import gymnasium as gym
import numpy as np
from typing import Any


class ActionAwareRewardWrapper(gym.Wrapper):
    """Algorithm-aware reward shaping for HighwayEnv experiments.

    The base HighwayEnv reward remains the main optimization target. This wrapper
    only adds small safety and smoothness terms that are meaningful for the
    current action space: categorical action labels are never subtracted, while
    continuous PPO actions may receive an L2 smoothness penalty.
    """

    def __init__(
        self,
        env: gym.Env,
        collision_penalty: float = 4.0,
        offroad_penalty: float = 2.0,
        discrete_action_penalty: float = 0.02,
        continuous_smooth_penalty: float = 0.05,
        target_speed: float = 27.0,
        speed_penalty: float = 0.01,
    ):
        super().__init__(env)
        self.collision_penalty = float(collision_penalty)
        self.offroad_penalty = float(offroad_penalty)
        self.discrete_action_penalty = float(discrete_action_penalty)
        self.continuous_smooth_penalty = float(continuous_smooth_penalty)
        self.target_speed = float(target_speed)
        self.speed_penalty = float(speed_penalty)
        self._last_continuous_action: np.ndarray | None = None

    def reset(self, **kwargs: Any):
        self._last_continuous_action = None
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = float(reward)

        crashed = bool(info.get('crashed', False)) or bool(getattr(getattr(self.unwrapped, 'vehicle', None), 'crashed', False))
        if crashed:
            shaped_reward -= self.collision_penalty

        vehicle = getattr(self.unwrapped, 'vehicle', None)
        if vehicle is not None:
            if not bool(getattr(vehicle, 'on_road', True)):
                shaped_reward -= self.offroad_penalty
            speed = getattr(vehicle, 'speed', None)
            if speed is not None:
                shaped_reward -= self.speed_penalty * abs(float(speed) - self.target_speed) / max(self.target_speed, 1.0)

        if isinstance(self.action_space, gym.spaces.Discrete):
            # Discrete actions are categories. Penalize non-idle decisions lightly,
            # but do not compute artificial distances between action labels.
            action_id = int(np.asarray(action).item())
            if action_id != 1:
                shaped_reward -= self.discrete_action_penalty

        elif isinstance(self.action_space, gym.spaces.Box):
            action_vec = np.asarray(action, dtype=np.float32).reshape(-1)
            if self._last_continuous_action is not None:
                delta = action_vec - self._last_continuous_action
                shaped_reward -= self.continuous_smooth_penalty * float(np.dot(delta, delta))
            self._last_continuous_action = action_vec.copy()

        return obs, shaped_reward, terminated, truncated, info
