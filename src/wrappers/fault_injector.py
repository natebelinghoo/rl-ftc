import gymnasium as gym
import numpy as np
from typing import Any


class FaultInjectorWrapper(gym.ActionWrapper):
    """故障注入中间件: 在动作传给物理引擎之前进行篡改。"""

    _FAULT_TYPES = ("none", "stuck", "gain_loss", "bias")

    def __init__(self, env: gym.Env, fault_type: str = "none", severity: float = 0.5):
        super().__init__(env)
        if fault_type not in (*self._FAULT_TYPES, "random"):
            raise ValueError(f"Unsupported fault_type: {fault_type}")
        self.fault_type = fault_type
        self.severity = float(np.clip(severity, 0.0, 1.0))
        self.current_fault_type = "none"
        self.current_severity = 0.0
        self._last_discrete_action: int | None = None
        self._last_continuous_action: np.ndarray | None = None
        self._set_fault_for_episode()

    def _available_fault_types(self) -> tuple[str, ...]:
        if isinstance(self.action_space, gym.spaces.Discrete):
            return ("none", "stuck")
        if isinstance(self.action_space, gym.spaces.Box):
            return self._FAULT_TYPES
        return ("none",)

    def _sample_severity(self, fault_type: str) -> float:
        if fault_type == "stuck":
            return float(np.random.uniform(0.05, 0.35))
        if fault_type == "gain_loss":
            return float(np.random.uniform(0.1, 0.5))
        if fault_type == "bias":
            return float(np.random.uniform(0.03, 0.2))
        return 0.0

    def _set_fault_for_episode(self) -> None:
        """在 reset 前确定本 episode 故障配置，random 会按动作空间采样有效故障。"""
        if self.fault_type == "random":
            sampled_fault = str(np.random.choice(self._available_fault_types()))
            self.current_fault_type = sampled_fault
            self.current_severity = self._sample_severity(sampled_fault)
            return

        if self.fault_type not in self._available_fault_types():
            self.current_fault_type = "none"
            self.current_severity = 0.0
            return

        self.current_fault_type = self.fault_type
        self.current_severity = 0.0 if self.fault_type == "none" else self.severity

    def get_fault_vector(self) -> np.ndarray:
        """返回长度为 5 的故障向量 [is_none, is_stuck, is_gain_loss, is_bias, severity]。"""
        one_hot = np.zeros(4, dtype=np.float32)
        type_idx = self._FAULT_TYPES.index(self.current_fault_type)
        one_hot[type_idx] = 1.0
        severity = np.array([self.current_severity], dtype=np.float32)
        return np.concatenate([one_hot, severity]).astype(np.float32)

    def reset(self, **kwargs: Any):
        self._set_fault_for_episode()
        self._last_discrete_action = None
        self._last_continuous_action = None
        return self.env.reset(**kwargs)

    def action(self, action: Any) -> Any:
        if self.current_fault_type == "none":
            if isinstance(self.action_space, gym.spaces.Discrete):
                self._last_discrete_action = int(np.asarray(action).item())
            elif isinstance(self.action_space, gym.spaces.Box):
                self._last_continuous_action = np.asarray(action, dtype=np.float32).copy()
            return action

        if isinstance(self.action_space, gym.spaces.Discrete):
            action_id = int(np.asarray(action).item())
            if (
                self.current_fault_type == "stuck"
                and self._last_discrete_action is not None
                and np.random.random() < self.current_severity
            ):
                return self._last_discrete_action
            self._last_discrete_action = action_id
            return action_id

        if isinstance(self.action_space, gym.spaces.Box):
            faulty_action = np.asarray(action, dtype=np.float32).copy()
            low = np.asarray(self.action_space.low, dtype=np.float32)
            high = np.asarray(self.action_space.high, dtype=np.float32)

            if self.current_fault_type == "stuck":
                if self._last_continuous_action is not None and np.random.random() < self.current_severity:
                    return self._last_continuous_action.copy()
                self._last_continuous_action = np.clip(faulty_action, low, high).astype(np.float32)
                return self._last_continuous_action.copy()

            if self.current_fault_type == "gain_loss":
                faulty_action[1] = faulty_action[1] * (1.0 - self.current_severity)
            elif self.current_fault_type == "bias":
                faulty_action[0] = faulty_action[0] + self.current_severity

            faulty_action = np.clip(faulty_action, low, high).astype(np.float32)
            self._last_continuous_action = faulty_action.copy()
            return faulty_action

        return action
