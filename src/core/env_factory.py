# src/core/env_factory.py: 负责组装 Env + Config + Wrapper
import gymnasium as gym
import highway_env  # noqa: F401 确保注册 highway-* 环境
from src.wrappers.fault_injector import FaultInjectorWrapper

def make_env(scenario_config: dict, fault_type: str = "none", fault_severity: float = 0.0, render_mode=None):
    """
    工厂函数：返回一个 Gym 环境实例
    """
    env_id = scenario_config["env_id"]
    config = scenario_config["config"]

    # 定义构建单个环境的函数
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        # gymnasium 会返回 OrderEnforcing 包装，直接调用 configure 会报错，需对底层 env 调用
        env.unwrapped.configure(config)
        env.reset()
        # 包装故障注入器
        env = FaultInjectorWrapper(env, fault_type=fault_type, severity=fault_severity)
        return env

    return _init()