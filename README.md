# Mountain Car Continuous - Reinforcement Learning Project

This is a complete reinforcement learning project for the Mountain Car Continuous environment using Gymnasium and Stable-Baselines3.

## Environment Description

**Mountain Car Continuous** is a classic control problem where:
- A car is placed at the bottom of a sinusoidal valley
- The goal is to reach the top of the right hill (position ≥ 0.45)
- The car can apply continuous directional forces in the range [-1, 1]
- Each timestep incurs a penalty of -0.1 × (action)²
- Reaching the goal gives a reward of +100

### Key Parameters
- **Observation Space**: Position [-1.2, 0.6] and Velocity [-0.07, 0.07]
- **Action Space**: Continuous force in [-1, 1]
- **Episode Horizon**: Max 999 timesteps

## Project Structure

```
mountain_car_continuous/
├── config.py              # Configuration and hyperparameters
├── train.py               # Training script
├── evaluate.py            # Evaluation and benchmarking
├── visualize.py           # Visualization tools
├── requirements.txt       # Python dependencies
├── models/                # Saved model checkpoints
├── logs/                  # TensorBoard logs
└── README.md             # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
# Navigate to project directory
cd mountain_car_continuous

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Hyperparameters

Edit `config.py` to adjust:
- **Algorithm**: "PPO", "DDPG", "TD3", or "SAC"
- **TOTAL_TIMESTEPS**: Training duration (default: 100,000)
- **LEARNING_RATE**: Optimizer learning rate (default: 3e-4)
- **BATCH_SIZE**: Batch size for training (default: 64)
- **N_STEPS**: Steps per rollout for PPO (default: 2048)

## Usage

### Training

Train a new agent:

```bash
python train.py
```

**What happens:**
- Creates vectorized environment (4 parallel environments)
- Initializes PPO agent with tuned hyperparameters
- Trains for specified timesteps
- Saves checkpoints periodically
- Evaluates on separate environment
- Saves final model

**Output:**
- `models/ppo_mountain_car_continuous_final.zip` - Final trained model
- `models/ppo_*.zip` - Checkpoint files
- `logs/` - TensorBoard logs

### Evaluation

Evaluate a trained model:

```bash
python evaluate.py
```

**Features:**
- Loads trained model
- Runs 10 evaluation episodes
- Reports statistics (mean reward, success rate, etc.)
- Renders 3 episodes with environment visualization
- Tracks performance metrics

### Visualization

Generate analysis plots:

```bash
python visualize.py
```

**Generates:**
- `trajectory_visualization.png` - Position, velocity, action, reward over time
- `policy_analysis.png` - Heatmap of policy across state space

## Expected Results

With the default configuration (100k steps), you should expect:
- **Success Rate**: 80-100% (reaching goal)
- **Mean Episode Length**: ~200-500 steps
- **Mean Reward**: Approximately -90 to -100 (after goal reward)

## Algorithm Details: PPO (Proximal Policy Optimization)

**Why PPO?**
- Combines the benefits of policy gradient and value-based methods
- Stable training with good sample efficiency
- State-of-the-art performance on continuous control tasks
- Robust to hyperparameter choices

**Key Hyperparameters:**
- `n_steps`: Trajectory rollout length (2048)
- `batch_size`: Mini-batch size for gradient updates (64)
- `n_epochs`: Number of passes over the collected data (10)
- `clip_range`: Clipping parameter for policy updates (0.2)
- `gamma`: Discount factor (0.9999)

## Advanced Usage

### Training with Different Algorithms

To use DDPG instead of PPO, modify `config.py`:

```python
ALGORITHM = "DDPG"
```

And update `train.py`:

```python
from stable_baselines3 import DDPG

model = DDPG("MlpPolicy", env, learning_rate=learning_rate, verbose=1)
```

### TensorBoard Monitoring

Monitor training in real-time:

```bash
tensorboard --logdir=logs/
```

Then open `http://localhost:6006` in your browser.

### Custom Evaluation

Create custom evaluation scripts by importing from the modules:

```python
from train import create_environment
from evaluate import load_model, evaluate_agent

model = load_model("models/ppo_mountain_car_continuous_final")
stats, rewards, lengths = evaluate_agent(model, n_episodes=20, render=True)
```

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` in config.py
- Reduce number of parallel environments in `train.py` (change `n_envs=4`)

### Slow Training
- Increase `n_envs` for more parallel environments
- Ensure GPU support for faster computation

### Model Not Improving
- Try increasing `TOTAL_TIMESTEPS`
- Adjust `LEARNING_RATE` (try 1e-4 or 5e-4)
- Increase `n_steps` for longer rollouts

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Mountain Car Original Paper](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)

## License

This project is for educational purposes.
