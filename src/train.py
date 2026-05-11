# src/train.py: 训练入口
import argparse
import os
import time
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from src.configs.scenarios import DQN_CONFIG, PPO_CONFIG
from src.core.env_factory import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["dqn", "ppo"], required=True, help="Choose algo")
    parser.add_argument("--steps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--fault_mode", type=str, choices=["none", "fixed", "random"], default="random")
    parser.add_argument("--fault", type=str, choices=["none", "stuck", "gain_loss", "bias"], default="none")
    parser.add_argument("--severity", type=float, default=0.0)
    parser.add_argument("--fault_aware", type=lambda value: value.lower() in {"true", "1", "yes", "y"}, default=True)
    parser.add_argument("--eval_best", type=lambda value: value.lower() in {"true", "1", "yes", "y"}, default=True)
    parser.add_argument("--load_model", type=str, default=None, help="Optional model path without .zip for continued training")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{args.algo}_aware_{str(args.fault_aware).lower()}"
    model_dir = f"outputs/models/{model_name}"
    log_dir = f"outputs/logs/{model_name}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)

    if args.fault_mode == "none":
        fault_type = "none"
        fault_severity = 0.0
    elif args.fault_mode == "fixed":
        fault_type = args.fault
        fault_severity = args.severity
    else:
        fault_type = "random"
        fault_severity = 0.0

    if args.algo == "dqn":
        config = DQN_CONFIG
        ModelClass = DQN
    else:
        config = PPO_CONFIG
        ModelClass = PPO

    total_steps = args.steps or int(config.get("default_steps", 50000))
    env = make_env(
        config,
        fault_type=fault_type,
        fault_severity=fault_severity,
        fault_aware=args.fault_aware,
    )
    env = Monitor(env, log_dir)

    print(
        f"Initializing {args.algo.upper()} model "
        f"(steps={total_steps}, fault_mode={args.fault_mode}, fault={fault_type}, "
        f"severity={fault_severity:.3f}, fault_aware={args.fault_aware})"
    )
    if args.load_model:
        print(f"Continuing from model: {args.load_model}")
        model = ModelClass.load(args.load_model, env=env, verbose=1, tensorboard_log=log_dir)
    else:
        model = ModelClass(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            **config["algo_params"],
        )

    callback = None
    if args.eval_best:
        eval_env = make_env(
            config,
            fault_type="none",
            fault_severity=0.0,
            fault_aware=args.fault_aware,
        )
        eval_env = Monitor(eval_env)
        callback = EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=5000,
            n_eval_episodes=20,
            deterministic=True,
            render=False,
            verbose=1,
        )

    print(f"Start training for {total_steps} steps...")
    model.learn(total_timesteps=total_steps, callback=callback, reset_num_timesteps=args.load_model is None)

    save_path = f"{model_dir}/final_model"
    model.save(save_path)
    print(f"Training finished. Model saved to: {save_path}")


if __name__ == "__main__":
    main()
