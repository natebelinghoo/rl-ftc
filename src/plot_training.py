# 绘制训练曲线（episode reward / length），读取 monitor.csv
import argparse
import glob
import os
import csv
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_monitor_files(log_dir: str) -> List[str]:
    pattern = os.path.join(log_dir, "monitor*.csv")
    return sorted(glob.glob(pattern))


def read_monitor_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    rewards = []
    lengths = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            # monitor.csv 格式: r, l, t
            try:
                r = float(row[0])
                l = float(row[1])
                rewards.append(r)
                lengths.append(l)
            except (ValueError, IndexError):
                continue
    return np.array(rewards, dtype=float), np.array(lengths, dtype=float)


def smooth(y: np.ndarray, window: int = 10) -> np.ndarray:
    if len(y) == 0:
        return y
    window = max(1, min(window, len(y)))
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def plot_curves(rewards: np.ndarray, lengths: np.ndarray, out_path: str, smooth_window: int = 20):
    plt.figure(figsize=(10, 5))

    # 奖励曲线
    plt.subplot(1, 2, 1)
    if len(rewards):
        plt.plot(rewards, alpha=0.3, label="Episode Reward (raw)")
        smoothed = smooth(rewards, smooth_window)
        plt.plot(np.arange(len(smoothed)) + smooth_window - 1, smoothed, label=f"Smoothed ({smooth_window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Episode Reward")
    plt.legend()
    plt.grid(True)

    # 时长曲线
    plt.subplot(1, 2, 2)
    if len(lengths):
        plt.plot(lengths, alpha=0.3, label="Episode Length (raw)")
        smoothed_len = smooth(lengths, smooth_window)
        plt.plot(np.arange(len(smoothed_len)) + smooth_window - 1, smoothed_len, label=f"Smoothed ({smooth_window})")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Training Episode Length")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="目录包含 monitor*.csv")
    parser.add_argument("--out", type=str, default="outputs/figures/training_curve.png", help="输出图片路径")
    parser.add_argument("--smooth", type=int, default=20, help="滑动平均窗口大小")
    args = parser.parse_args()

    files = load_monitor_files(args.log_dir)
    if not files:
        raise FileNotFoundError(f"No monitor*.csv found in {args.log_dir}")

    all_rewards = []
    all_lengths = []
    for fpath in files:
        r, l = read_monitor_csv(fpath)
        all_rewards.append(r)
        all_lengths.append(l)

    rewards = np.concatenate(all_rewards) if all_rewards else np.array([])
    lengths = np.concatenate(all_lengths) if all_lengths else np.array([])

    plot_curves(rewards, lengths, args.out, smooth_window=args.smooth)


if __name__ == "__main__":
    main()

