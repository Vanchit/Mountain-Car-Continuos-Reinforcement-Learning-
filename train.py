"""Training script for Mountain Car Continuous using Stable-Baselines3."""

import gymnasium as gym
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from pathlib import Path

import config


def create_environment(render_mode=None):
    """Create and return the Mountain Car Continuous environment."""
    env = gym.make(config.ENV_ID, render_mode=render_mode, goal_velocity=config.GOAL_VELOCITY)
    return env


def train_agent(algorithm="SAC", total_timesteps=None, learning_rate=None):
    """
    Train an RL agent on Mountain Car Continuous using reinforcement learning.
    
    Args:
        algorithm (str): Algorithm to use (e.g., "SAC", "DDPG", "PPO")
        total_timesteps (int): Total timesteps for training
        learning_rate (float): Learning rate for the optimizer
    
    Returns:
        Trained model and environment
    """
    if total_timesteps is None:
        total_timesteps = config.TOTAL_TIMESTEPS
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    
    print(f"Creating Mountain Car Continuous environment: {config.ENV_ID}")
    env = gym.make(config.ENV_ID, render_mode=None)
    
    print(f"Initializing {algorithm} agent for Mountain Car...")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Total Timesteps: {total_timesteps:,}")
    
    # SAC with strong entropy for exploration on Mountain Car
    if algorithm.upper() == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=100000,
            learning_starts=100,  # Start learning immediately  
            batch_size=config.BATCH_SIZE,
            ent_coef="auto_0.1",  # Auto entropy with higher weight for exploration
            target_update_interval=1,
            gamma=0.999,  # Very high discount for long-term horizons
            tau=0.01,  # Slower target updates
            verbose=1,
            tensorboard_log=str(config.LOGS_DIR)
        )
    else:
        model = DDPG(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=100000,
            learning_starts=100,
            batch_size=64,
            gamma=0.995,
            tau=0.005,
            train_freq=(1, "step"),
            verbose=1,
            tensorboard_log=str(config.LOGS_DIR)
        )
    
    # Callback to save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=config.SAVE_FREQ,
        save_path=str(config.MODELS_DIR),
        name_prefix=config.ALGORITHM.lower()
    )
    
    # Callback to evaluate model periodically
    eval_callback = EvalCallback(
        gym.make(config.ENV_ID),
        best_model_save_path=str(config.MODELS_DIR),
        log_path=str(config.LOGS_DIR),
        eval_freq=5000,
        deterministic=config.DETERMINISTIC_EVAL,
        render=False
    )
    
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback]
    )
    
    # Save final model
    final_model_path = config.MODELS_DIR / f"{config.MODEL_NAME}_final"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}.zip")
    
    env.close()
    return model


def main():
    """Main training entry point."""
    print("=" * 60)
    print("Mountain Car Continuous - RL Training")
    print("=" * 60)
    
    model = train_agent(
        algorithm=config.ALGORITHM,
        total_timesteps=config.TOTAL_TIMESTEPS,
        learning_rate=config.LEARNING_RATE
    )
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Models saved in: {config.MODELS_DIR}")
    print(f"Logs saved in: {config.LOGS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
