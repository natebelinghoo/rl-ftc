# 1. 训练 DQN
python src/train.py --algo dqn  --steps 30000

# 2. 训练 PPO
python src/train.py --algo ppo

# 3. 评估 - 生成正常视频
python src/evaluate.py --algo ppo --fault none

# 4. 评估 - 生成故障视频
# PPO 遇到方向盘跑偏故障
python src/evaluate.py --algo ppo --fault bias 
# DQN 遇到无法变道故障
python src/evaluate.py --algo dqn --fault stuck