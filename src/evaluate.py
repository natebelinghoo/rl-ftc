# src/evaluate.py: 评估入口
import argparse
import os
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from src.configs.scenarios import DQN_CONFIG, PPO_CONFIG
from src.core.env_factory import make_env


def str2bool(value: str) -> bool:
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("fault_aware must be true/false")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["dqn", "ppo"], required=True)
    parser.add_argument("--fault", type=str, default="none", help="none, gain_loss, bias, stuck")
    parser.add_argument("--fault_aware", type=str2bool, default=True, help="Enable fault-aware observation")
    args = parser.parse_args()

    # 1. 配置
    if args.algo == "dqn":
        config = DQN_CONFIG
        ModelClass = DQN
        severity = 0.3 # 30%概率卡死
    else:
        config = PPO_CONFIG
        ModelClass = PPO
        severity = 0.2 if args.fault == "bias" else 0.5 # 偏置0.2 或 动力损失50%

    print(
        f"🧪 Evaluation config: algo={args.algo}, fault={args.fault}, "
        f"severity={severity:.3f}, fault_aware={args.fault_aware}"
    )

    # 2. 准备带录像功能的向量化环境
    video_folder = f"outputs/videos/{args.algo}_{args.fault}"
    os.makedirs(video_folder, exist_ok=True)

    def create_env_fn():
        # 这里必须指定 render_mode='rgb_array' 才能录像
        return make_env(
            config,
            fault_type=args.fault,
            fault_severity=severity,
            render_mode="rgb_array",
            fault_aware=args.fault_aware,
        )

    # 包装成 VecEnv
    vec_env = DummyVecEnv([create_env_fn])
    
    # 包装 VideoRecorder
    vec_env = VecVideoRecorder(
        vec_env, 
        video_folder, 
        record_video_trigger=lambda x: x == 0, # 第一次reset就录像
        video_length=300, # 录制 300 帧
        name_prefix=f"result"
    )

    # 3. 加载模型
    model_path = f"outputs/models/{args.algo}/final_model"
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ Model not found at {model_path}. Run train.py first!")
        return

    print(f"📥 Loading model from {model_path}...")
    model = ModelClass.load(model_path, env=vec_env)

    # 4. 运行推理 (录像会自动保存)
    print("🎬 Running simulation...")
    obs = vec_env.reset()
    for _ in range(300 + 10):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = vec_env.step(action)
    
    vec_env.close()
    print(f"✨ Video saved to {video_folder}")

if __name__ == "__main__":
    main()
