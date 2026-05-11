# src/evaluate.py: 评估入口
import argparse
import json
import os
import time
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


def summarize(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    return mean, min(values), max(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["dqn", "ppo"], required=True)
    parser.add_argument("--fault", type=str, default="none", help="none, gain_loss, bias, stuck")
    parser.add_argument("--fault_aware", type=str2bool, default=True, help="Enable fault-aware observation")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--record_video", type=str2bool, default=False, help="Record one evaluation video")
    parser.add_argument("--model_name", type=str, default=None, help="Model folder under outputs/models")
    args = parser.parse_args()

    if args.algo == "dqn" and args.fault not in {"none", "stuck"}:
        raise SystemExit("DQN uses a discrete meta-action space; only none/stuck faults are valid. Use PPO for gain_loss or bias.")

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
        f"severity={severity:.3f}, fault_aware={args.fault_aware}, "
        f"episodes={args.episodes}, max_steps={args.max_steps}"
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
    
    # 包装 VideoRecorder（可选）
    if args.record_video:
        vec_env = VecVideoRecorder(
            vec_env,
            video_folder,
            record_video_trigger=lambda x: x == 0,  # 第一次reset就录像
            video_length=args.max_steps,
            name_prefix="result",
        )

    # 3. 加载模型
    preferred_model = args.model_name or f"{args.algo}_aware_{str(args.fault_aware).lower()}"
    model_path = f"outputs/models/{preferred_model}/final_model"
    legacy_model_path = f"outputs/models/{args.algo}/final_model"
    if not os.path.exists(model_path + ".zip") and os.path.exists(legacy_model_path + ".zip"):
        model_path = legacy_model_path
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ Model not found at {model_path}. Run train.py first!")
        return

    print(f"📥 Loading model from {model_path}...")
    try:
        model = ModelClass.load(model_path, env=vec_env)
    except ValueError as exc:
        print("❌ Failed to load model due to space mismatch.")
        print(f"   detail: {exc}")
        print("   请确认训练与评估使用一致的 observation/action 配置（包括 --fault_aware 与场景配置）。")
        print("   如果当前代码已改动，通常需要重新训练对应模型。")
        vec_env.close()
        return

    # 4. 运行评估（可录像 + 定量指标）
    print("🎬 Running evaluation...")
    episode_rewards: list[float] = []
    episode_lengths: list[float] = []
    crash_flags: list[float] = []
    safe_finish_flags: list[float] = []

    for _ in range(args.episodes):
        obs = vec_env.reset()
        ep_reward = 0.0
        crashed = False
        ep_len = 0

        for _ in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = vec_env.step(action)

            ep_reward += float(reward[0])
            ep_len += 1
            crashed = crashed or bool(infos[0].get("crashed", False))
            if bool(done[0]):
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(float(ep_len))
        crash_flags.append(1.0 if crashed else 0.0)
        safe_finish_flags.append(0.0 if crashed else 1.0)

    vec_env.close()

    avg_reward, min_reward, max_reward = summarize(episode_rewards)
    avg_length, _, _ = summarize(episode_lengths)
    crash_rate, _, _ = summarize(crash_flags)
    safe_finish_rate, _, _ = summarize(safe_finish_flags)

    print("📊 Evaluation metrics:")
    print(f"  avg_episode_reward: {avg_reward:.3f} (min={min_reward:.3f}, max={max_reward:.3f})")
    print(f"  avg_episode_length: {avg_length:.1f}")
    print(f"  crash_rate: {crash_rate:.3f}")
    print(f"  safe_finish_rate: {safe_finish_rate:.3f}")

    metrics_dir = "outputs/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    metrics_path = (
        f"{metrics_dir}/eval_{args.algo}_fault-{args.fault}_aware-{str(args.fault_aware).lower()}_{ts}.json"
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "algo": args.algo,
                "fault": args.fault,
                "severity": severity,
                "fault_aware": args.fault_aware,
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "avg_episode_reward": avg_reward,
                "min_episode_reward": min_reward,
                "max_episode_reward": max_reward,
                "avg_episode_length": avg_length,
                "crash_rate": crash_rate,
                "safe_finish_rate": safe_finish_rate,
                "episode_rewards": episode_rewards,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"📝 Metrics saved to {metrics_path}")
    if args.record_video:
        print(f"✨ Video saved to {video_folder}")

if __name__ == "__main__":
    main()
