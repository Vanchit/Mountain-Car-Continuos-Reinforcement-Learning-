# Mountain Car Continuous - Reinforcement Learning

**An agent learning to climb a mountain using Reinforcement Learning (SAC algorithm).**

## 🚗 What Is This?

A trained AI agent that learns to drive a car up a steep hill using **Soft Actor-Critic (SAC)** reinforcement learning algorithm. The car:
- Starts at the bottom of a valley (position = -1.2)
- Cannot climb directly (weak engine)
- **Learns through trial and error** to swing back and forth
- **Reaches the goal** at the top (position ≥ 0.45)

**Reference:** https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/

## 🎯 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the RL agent (learns to climb mountain)
python train.py

# Evaluate performance
python evaluate.py

# Visualize results
python visualize.py
```

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `train.py` | Trains the RL agent using SAC algorithm |
| `evaluate.py` | Tests and benchmarks the trained agent |
| `visualize.py` | Shows trajectory plots and policy analysis |
| `config.py` | Configuration & hyperparameters |
| `models/` | Saved trained models |
| `logs/` | Training metrics & TensorBoard logs |

## 🤖 Algorithm: SAC (Soft Actor-Critic)

**Why SAC?**
- Excellent exploration with entropy regularization
- Stable training for continuous control
- Good for Mountain Car problem
- Learns optimal swing strategy automatically

**Training Details:**
- Timesteps: 50,000 (fast training)
- Learning Rate: 2e-3
- Batch Size: 128
- Evaluation Episodes: 5

## 📊 Output Files Generated

After training and visualization:
- `models/best_model.zip` - Trained agent
- `trajectory_visualization.png` - Position, velocity, action, reward plots
- `policy_analysis.png` - Heatmap of learned policy

## 🔗 References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [SAC Algorithm Paper](https://arxiv.org/abs/1801.01290)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
