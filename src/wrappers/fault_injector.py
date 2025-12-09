# src/wrappers/fault_injector.py
import gymnasium as gym
import numpy as np

class FaultInjectorWrapper(gym.ActionWrapper):
    """
    故障注入中间件：在动作传给物理引擎之前进行篡改
    """
    def __init__(self, env, fault_type: str = "none", severity: float = 0.5):
        super().__init__(env)
        self.fault_type = fault_type
        self.severity = severity 
        
    def action(self, action):
        if self.fault_type == "none":
            return action

        # --- DQN 场景 (离散动作) ---
        # 动作是整数索引，映射到元动作
        if isinstance(self.action_space, gym.spaces.Discrete):
            if self.fault_type == "stuck":
                # 30%概率卡死（强制保持当前车道，即动作1-LANE_KEEP）
                # 注意：highway-env中 0:LANE_LEFT, 1:IDLE, 2:LANE_RIGHT
                if np.random.random() < self.severity:
                    return 1 
            
        # --- PPO 场景 (连续动作) ---
        # 动作是 [steering, acceleration]
        elif isinstance(self.action_space, gym.spaces.Box):
            # 必须 copy，防止修改引用
            faulty_action = action.copy()
            
            if self.fault_type == "gain_loss":
                # 动力衰减：油门(索引1)打折
                # highway-env的油门范围通常是[-1, 1]
                faulty_action[1] = faulty_action[1] * (1.0 - self.severity)
                
            elif self.fault_type == "bias":
                # 方向跑偏：转向(索引0)增加偏差
                faulty_action[0] = faulty_action[0] + self.severity
            
            return faulty_action

        return action