"""Configuration file for Mountain Car Continuous training."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Environment configuration
ENV_ID = "MountainCarContinuous-v0"
RENDER_MODE = None  # Set to "rgb_array" or "human" for visualization

# Training configuration
ALGORITHM = "SAC"  # Options: "PPO", "DDPG", "TD3", "SAC" - SAC with entropy works well
TOTAL_TIMESTEPS = 200000  # Mountain Car requires focused training
LEARNING_RATE = 1e-3  # Good learning rate
BATCH_SIZE = 64  # Batch size for SAC
N_STEPS = 4096  # For reference (PPO uses this)

# Environment-specific parameters
GOAL_VELOCITY = 0.0
MAX_EPISODE_STEPS = 999

# Evaluation configuration
EVAL_EPISODES = 10
DETERMINISTIC_EVAL = True

# Model saving
MODEL_NAME = f"{ALGORITHM}_mountain_car_continuous"
MODEL_PATH = MODELS_DIR / MODEL_NAME
SAVE_FREQ = 10000  # Save model every N timesteps
