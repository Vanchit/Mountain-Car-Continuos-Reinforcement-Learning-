"""Evaluation script for trained Mountain Car Continuous agents."""

import gymnasium as gym
from stable_baselines3 import PPO, SAC, DDPG
import numpy as np
from pathlib import Path

import config


def load_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str or Path): Path to the saved model (without .zip extension)
    
    Returns:
        Loaded model
    """
    model_path = Path(model_path)
    if not model_path.suffix == ".zip":
        model_path = model_path.with_suffix(".zip")
    
    print(f"Loading model from: {model_path}")
    
    # Auto-detect algorithm from filename or config
    algo_lower = str(model_path).lower()
    if "ddpg" in algo_lower:
        model = DDPG.load(str(model_path))
    elif "sac" in algo_lower:
        model = SAC.load(str(model_path))
    else:
        model = PPO.load(str(model_path))
    
    return model


def evaluate_agent(model, n_episodes=None, render=False, deterministic=True):
    """
    Evaluate a trained agent on the Mountain Car Continuous environment.
    
    Args:
        model: Trained RL model
        n_episodes (int): Number of episodes to run
        render (bool): Whether to render the environment
        deterministic (bool): Whether to use deterministic (greedy) actions
    
    Returns:
        Dictionary with evaluation statistics
    """
    if n_episodes is None:
        n_episodes = config.EVAL_EPISODES
    
    env = gym.make(config.ENV_ID, render_mode="human" if render else None)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"\nEvaluating agent for {n_episodes} episodes...")
    print(f"Deterministic: {deterministic}, Render: {render}")
    
    for episode in range(n_episodes):
        observation, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Get action from model
            action, _ = model.predict(observation, deterministic=deterministic)
            
            # Execute action
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        # Success if goal was reached (position >= 0.45) before truncation
        success = terminated and not truncated
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if success:
            success_count += 1
        
        status = "✓ SUCCESS" if success else "✗ TIMEOUT"
        print(f"Episode {episode + 1}/{n_episodes}: Reward={episode_reward:7.2f}, "
              f"Length={episode_length}, {status}")
    
    env.close()
    
    # Calculate statistics
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "mean_episode_length": np.mean(episode_lengths),
        "success_rate": success_count / n_episodes,
        "total_successes": success_count
    }
    
    return stats, episode_rewards, episode_lengths


def print_statistics(stats):
    """Pretty print evaluation statistics."""
    print("\n" + "=" * 60)
    print("Evaluation Statistics")
    print("=" * 60)
    print(f"Mean Reward:          {stats['mean_reward']:8.2f}")
    print(f"Std Reward:           {stats['std_reward']:8.2f}")
    print(f"Max Reward:           {stats['max_reward']:8.2f}")
    print(f"Min Reward:           {stats['min_reward']:8.2f}")
    print(f"Mean Episode Length:  {stats['mean_episode_length']:8.1f}")
    print(f"Success Rate:         {stats['success_rate']:8.1%}")
    print(f"Total Successes:      {stats['total_successes']:8d}")
    print("=" * 60)


def main():
    """Main evaluation entry point."""
    print("=" * 60)
    print("Mountain Car Continuous - Evaluation")
    print("=" * 60)
    
    # Try to load the final model
    model_path = config.MODELS_DIR / f"{config.MODEL_NAME}_final"
    
    if not (Path(model_path).with_suffix(".zip")).exists():
        print(f"Error: Model not found at {model_path}.zip")
        print(f"Please train a model first using train.py")
        return
    
    model = load_model(model_path)
    
    # Evaluate without rendering
    stats, rewards, lengths = evaluate_agent(
        model,
        n_episodes=config.EVAL_EPISODES,
        render=False,
        deterministic=config.DETERMINISTIC_EVAL
    )
    
    print_statistics(stats)
    
    # Optionally render a few episodes
    print("\nRunning 3 episodes with visualization...")
    render_stats, _, _ = evaluate_agent(
        model,
        n_episodes=3,
        render=True,
        deterministic=True
    )


if __name__ == "__main__":
    main()
