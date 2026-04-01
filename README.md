# Mountain Car Continuous - Reinforcement Learning

A reinforcement learning project for solving Mountain Car Continuous using Stable-Baselines3 and Gymnasium.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train agent
python train.py

# Evaluate
python evaluate.py

# Visualize results
python visualize.py
```

## Project Files

- `config.py` - Hyperparameters and settings
- `train.py` - Training script
- `evaluate.py` - Model evaluation
- `visualize.py` - Result visualization
- `models/` - Trained models
- `logs/` - TensorBoard logs

## Problem Description

**Goal**: Drive a car up a sinusoidal valley to reach the top right hill (position ≥ 0.45)

**State Space**: Position [-1.2, 0.6] and Velocity [-0.07, 0.07]

**Action Space**: Continuous force in [-1, 1]

**Expected Results**: 80-100% success rate with ~200-500 steps per episode

## Configuration

Edit `config.py` to adjust:
- Algorithm: "SAC", "PPO", "DDPG"
- Training timesteps
- Learning rate
- Batch size



## References

- [Gymnasium Docs](https://gymnasium.farama.org/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
