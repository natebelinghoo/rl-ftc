# src/train.py: 训练入口
import argparse
import os
import time
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from src.configs.scenarios import DQN_CONFIG, PPO_CONFIG
from src.core.env_factory import make_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["dqn", "ppo"], required=True, help="Choose algo")
    parser.add_argument("--steps", type=int, default=50000, help="Total training timesteps")
    args = parser.parse_args()

    # 1. 准备目录
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_dir = f"outputs/models/{args.algo}"
    log_dir = f"outputs/logs/{args.algo}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)

    # 2. 加载配置
    if args.algo == "dqn":
        config = DQN_CONFIG
        ModelClass = DQN
        # DQN 训练时我们先不加故障，让它学会正常开
        env = make_env(config, fault_type="none")
    else:
        config = PPO_CONFIG
        ModelClass = PPO
        env = make_env(config, fault_type="none")

    # 3. 初始化模型
    env = Monitor(env, log_dir)
    print(f"🔥 Initializing {args.algo.upper()} model...")
    model = ModelClass(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        **config["algo_params"]
    )

    # 4. 训练
    print(f"🚀 Start training for {args.steps} steps...")
    model.learn(total_timesteps=args.steps)
    
    # 5. 保存
    save_path = f"{model_dir}/final_model"
    model.save(save_path)
    print(f"✅ Training finished. Model saved to: {save_path}")

if __name__ == "__main__":
    main()