# Mountain Car Continuous - Reinforcement Learning

**An agent learning to climb a mountain using Reinforcement Learning (RL) techniques.**

## 🎯 Project Goal

Train an AI agent to autonomously solve the **Mountain Car Continuous** problem - driving a car up a steep hill by learning optimal strategies through trial and error, without explicit programming.

## 🚗 What Is Mountain Car?

- **Environment**: A car at the bottom of a sinusoidal valley
- **Challenge**: Engine is too weak to climb directly - must swing back and forth
- **Goal**: Reach the top of the right hill (position ≥ 0.45)  
- **Learning**: Agent discovers the swing strategy through RL

## 🤖 How It Works

1. **Agent**: Uses SAC (Soft Actor-Critic) algorithm
2. **Learning**: Trained on 200,000 timesteps via trial and error
3. **Strategy**: Learns to swing left-right to build momentum
4. **Result**: Autonomous climbing behavior

## ⚡ Quick Start

```bash
pip install -r requirements.txt

# Train agent to climb mountain
python train.py

# See how it performs
python demo.py

# Watch it in action
python animate.py

# Analyze with plots
python visualize.py
```

## 📊 Project Structure

- `train.py` - Training loop
- `demo.py` - Live demonstration
- `evaluate.py` - Performance testing
- `visualize.py` - Result plots
- `config.py` - Settings
- `models/` - Trained agents
- `logs/` - Training metrics
