# 训练与推理调用封装
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from src.envs.tracking_env import TrackingEnv
from src.faults.actuator_fault import ActuatorFault
import numpy as np

class DDPGRunner:
    def __init__(self, fault: ActuatorFault | None = None):
        self.env = TrackingEnv()
        self.fault = fault

    def train(self):
        n_actions = self.env.action_space.shape[-1]
        noise = NormalActionNoise(np.zeros(n_actions), 0.1 * np.ones(n_actions))
        
        model = DDPG(
            "MlpPolicy",
            self.env,
            action_noise=noise,
            learning_rate=1e-3,
            verbose=1,
        )
        model.learn(20000)
        return model

    def rollout(self, model):
        obs = self.env.reset()
        xs, refs = [], []

        for t in range(200):
            action, _ = model.predict(obs)

            # 故障注入（如果有）
            if self.fault:
                action = np.array([self.fault.apply(action[0], t)])

            obs, _, done, _ = self.env.step(action)
            xs.append(obs[0])
            refs.append(obs[2])
            if done:
                break

        return xs, refs
