# 训练 DQN（离散动作，仅采样 none/stuck 有效故障）
python src/train.py --algo dqn

# 训练 PPO（连续动作，采样 none/stuck/gain_loss/bias）
python src/train.py --algo ppo

# 训练未增强观测基线
python src/train.py --algo dqn --fault_aware false
python src/train.py --algo ppo --fault_aware false

# 评估
python src/evaluate.py --algo dqn --fault none
python src/evaluate.py --algo dqn --fault stuck
python src/evaluate.py --algo ppo --fault none
python src/evaluate.py --algo ppo --fault bias
python src/evaluate.py --algo ppo --fault gain_loss
python src/evaluate.py --algo ppo --fault stuck

# 如需生成视频，显式打开 record_video
python src/evaluate.py --algo ppo --fault bias --record_video true
