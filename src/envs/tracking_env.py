# 系统动力学模型
import gym
from gym import spaces
import numpy as np

class TrackingEnv(gym.Env):
    def __init__(self, dt=0.1):
        super().__init__()
        self.dt = dt

        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-2.0, high=2.0, shape=(1,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.t = 0
        self.x = np.random.uniform(-1, 1)
        self.v = 0.0
        return self._get_obs()
    
    def _get_obs(self):
        ref = np.sin(0.1 * self.t)
        return np.array([self.x, self.v, ref], dtype=np.float32)
    
    def step(self, u):
        # 系统动力学(无故障)
        self.v += u[0] * self.dt
        self.x += self.v * self.dt
        ref = np.sin(0.1 * self.t)
        err = self.x - ref
        reward = - (err**2 + 0.1 * u[0]**2)
        
        self.t += 1
        done = self.t >= 200
        return self._get_obs(), reward, done, {}
