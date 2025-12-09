# src/configs/scenarios.py: 定义 DQN 和 PPO 的参数配置
import numpy as np

# DQN (离散动作 - 变道调度)
DQN_CONFIG = {
    "env_id": "highway-fast-v0",
    "config": {
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": 4,
        "vehicles_count": 30, 
        "duration": 40,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted"
        }
    },
    "algo_params": {
        "learning_rate": 5e-4,
        "buffer_size": 15000,
        "learning_starts": 200,
        "batch_size": 32,
        "gamma": 0.8,
        "train_freq": 1,
        "gradient_steps": 1,
        "target_update_interval": 50,
    }
}

# PPO (连续动作 - 动力学控制)
PPO_CONFIG = {
    "env_id": "highway-fast-v0", 
    "config": {
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
        },
        "lanes_count": 4,
        "vehicles_count": 10,
        "duration": 40,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        }
    },
    "algo_params": {
        "learning_rate": 5e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.90,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
    }
}