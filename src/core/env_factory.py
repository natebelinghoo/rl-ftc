# src/core/env_factory.py: 负责组装 Env + Config + Wrapper
import gymnasium as gym
import highway_env  # noqa: F401 确保注册 highway-* 环境
from src.wrappers.fault_aware_obs import FaultAwareObservationWrapper
from src.wrappers.fault_injector import FaultInjectorWrapper
from src.wrappers.reward_shaping import ActionAwareRewardWrapper


def make_env(
    scenario_config: dict,
    fault_type: str = "none",
    fault_severity: float = 0.0,
    render_mode=None,
    fault_aware: bool = True,
):
    """工厂函数：返回一个 Gym 环境实例。"""
    env_id = scenario_config["env_id"]
    config = scenario_config["config"]
    reward_config = scenario_config.get("reward", {})

    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env.unwrapped.configure(config)
        env.reset()
        env = FaultInjectorWrapper(env, fault_type=fault_type, severity=fault_severity)
        env = ActionAwareRewardWrapper(env, **reward_config)
        if fault_aware:
            env = FaultAwareObservationWrapper(env)
        return env

    return _init()
