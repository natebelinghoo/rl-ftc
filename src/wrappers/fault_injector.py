import gymnasium as gym
import numpy as np
from typing import Any


class FaultInjectorWrapper(gym.ActionWrapper):
    """
    故障注入中间件：在动作传给物理引擎之前进行篡改
    """

    _FAULT_TYPES = ("none", "stuck", "gain_loss", "bias")

    def __init__(self, env: gym.Env, fault_type: str = "none", severity: float = 0.5):
        super().__init__(env)
        if fault_type not in (*self._FAULT_TYPES, "random"):
            raise ValueError(f"Unsupported fault_type: {fault_type}")
        self.fault_type = fault_type
        self.severity = float(np.clip(severity, 0.0, 1.0))
        # current_* 代表当前 episode 实际生效的故障配置
        self.current_fault_type = "none"
        self.current_severity = 0.0
        self._set_fault_for_episode()

    def _set_fault_for_episode(self) -> None:
        """在 reset 前确定本 episode 故障配置（支持 fixed/random）。"""
        if self.fault_type == "random":
            sampled_fault = np.random.choice(self._FAULT_TYPES)
            if sampled_fault == "stuck":
                sampled_severity = float(np.random.uniform(0.1, 0.5))
            elif sampled_fault == "gain_loss":
                sampled_severity = float(np.random.uniform(0.1, 0.6))
            elif sampled_fault == "bias":
                sampled_severity = float(np.random.uniform(0.05, 0.3))
            else:
                sampled_severity = 0.0
            self.current_fault_type = sampled_fault
            self.current_severity = sampled_severity
            return

        self.current_fault_type = self.fault_type
        self.current_severity = 0.0 if self.fault_type == "none" else self.severity

    def get_fault_vector(self) -> np.ndarray:
        """
        返回长度为 5 的故障向量:
        [is_none, is_stuck, is_gain_loss, is_bias, severity]
        """
        one_hot = np.zeros(4, dtype=np.float32)
        type_idx = self._FAULT_TYPES.index(self.current_fault_type)
        one_hot[type_idx] = 1.0
        severity = np.array([self.current_severity], dtype=np.float32)
        return np.concatenate([one_hot, severity]).astype(np.float32)

    def reset(self, **kwargs: Any):
        # 每次 reset 都重新采样（仅 random）或恢复固定配置（非 random）
        self._set_fault_for_episode()
        return self.env.reset(**kwargs)

    def action(self, action: Any) -> Any:
        if self.current_fault_type == "none":
            return action

        # --- DQN 场景 (离散动作) ---
        # 动作是整数索引，映射到元动作
        if isinstance(self.action_space, gym.spaces.Discrete):
            if self.current_fault_type == "stuck":
                # 30%概率卡死（强制保持当前车道，即动作1-LANE_KEEP）
                # 注意：highway-env中 0:LANE_LEFT, 1:IDLE, 2:LANE_RIGHT
                if np.random.random() < self.current_severity:
                    return 1

        # --- PPO 场景 (连续动作) ---
        # 动作是 [steering, acceleration]
        elif isinstance(self.action_space, gym.spaces.Box):
            # 必须 copy，防止修改引用
            faulty_action = np.asarray(action, dtype=np.float32).copy()

            if self.current_fault_type == "gain_loss":
                # 动力衰减：油门(索引1)打折
                # highway-env的油门范围通常是[-1, 1]
                faulty_action[1] = faulty_action[1] * (1.0 - self.current_severity)

            elif self.current_fault_type == "bias":
                # 方向跑偏：转向(索引0)增加偏差
                faulty_action[0] = faulty_action[0] + self.current_severity

            return faulty_action

        return action
