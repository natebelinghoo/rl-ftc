# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement Learning based Fault-Tolerant Control using stable-baselines3 (DQN, PPO), gymnasium, and highway-env. Trains RL agents to control vehicles under actuator fault conditions (steering bias, power loss, stuck actions).

## Common Commands

```bash
# Training
python src/train.py --algo dqn --steps 30000
python src/train.py --algo ppo --steps 50000
python src/train.py --algo ppo --fault_mode random  # fault-agnostic training

# Evaluation (generates videos)
python src/evaluate.py --algo ppo --fault none
python src/evaluate.py --algo ppo --fault bias
python src/evaluate.py --algo dqn --fault stuck

# Plot training curves
python src/plot_training.py --log_dir outputs/logs/ppo_TIMESTAMP
```

## Architecture

**Entry Points:** `src/train.py`, `src/evaluate.py`

**Environment Factory (`src/core/env_factory.py`):**
- Creates `highway-fast-v0` environments
- Chains two wrappers: `FaultInjectorWrapper` → `FaultAwareObservationWrapper`

**Fault Injection (`src/wrappers/fault_injector.py`):**
- `gym.ActionWrapper` that corrupts actions before they reach the physics engine
- Fault types: `none`, `stuck` (discrete lane-keep), `gain_loss` (continuous throttle reduction), `bias` (continuous steering offset)
- Each `reset()` resamples fault if in `random` mode

**Fault-Aware Observation (`src/wrappers/fault_aware_obs.py`):**
- `gym.ObservationWrapper` that appends a 5-element fault vector `[is_none, is_stuck, is_gain_loss, is_bias, severity]` to the flattened observation
- Uses `find_wrapper()` helper to safely traverse the wrapper chain

**Scenario Configs (`src/configs/scenarios.py`):**
- `DQN_CONFIG`: DiscreteMetaAction (3 actions: left/idle/right), for lane-change scheduling
- `PPO_CONFIG`: ContinuousAction (steering + acceleration), for dynamics control

**Output Structure:** `outputs/models/{algo}/final_model.zip`, `outputs/logs/{algo}_{timestamp}/`, `outputs/videos/{algo}_{fault}/`