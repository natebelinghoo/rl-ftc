import gymnasium as gym
import numpy as np
from typing import Optional, Type, TypeVar

from src.wrappers.fault_injector import FaultInjectorWrapper

TWrapper = TypeVar("TWrapper", bound=gym.Wrapper)


def find_wrapper(env: gym.Env, wrapper_cls: Type[TWrapper]) -> Optional[TWrapper]:
    """
    安全地沿 wrapper 链向下查找指定类型, 避免直接写死 env.env.env 的脆弱访问
    """
    current = env
    while isinstance(current, gym.Wrapper):
        if isinstance(current, wrapper_cls):
            return current
        current = current.env
    if isinstance(current, wrapper_cls):
        return current
    return None


class FaultAwareObservationWrapper(gym.ObservationWrapper):
    """
    Fault-aware observation 核心实现: 
    将原始观测 flatten 后，拼接故障元信息向量 [one-hot fault type + severity]
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        injector = find_wrapper(self.env, FaultInjectorWrapper)
        if injector is None:
            raise RuntimeError("FaultAwareObservationWrapper requires FaultInjectorWrapper in env chain.")
        self._fault_injector: FaultInjectorWrapper = injector

        if not isinstance(self.observation_space, gym.spaces.Box):
            raise TypeError("FaultAwareObservationWrapper currently supports Box observation space only.")

        obs_low = np.asarray(self.observation_space.low, dtype=np.float32).reshape(-1)
        obs_high = np.asarray(self.observation_space.high, dtype=np.float32).reshape(-1)

        # fault vector: [is_none, is_stuck, is_gain_loss, is_bias, severity]
        fault_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        fault_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        low = np.concatenate([obs_low, fault_low]).astype(np.float32)
        high = np.concatenate([obs_high, fault_high]).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        flat_obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        fault_vec = self._fault_injector.get_fault_vector()
        obs_aug = np.concatenate([flat_obs, fault_vec]).astype(np.float32)
        return obs_aug
