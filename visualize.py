"""Visualization script for Mountain Car Continuous training analysis."""

import gymnasium as gym
from stable_baselines3 import PPO, SAC, DDPG
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import config


def plot_episode_trajectory(model, episode_num=1, deterministic=True):
    """
    Plot the trajectory of a single episode.
    
    Args:
        model: Trained RL model
        episode_num (int): Episode number (for title)
        deterministic (bool): Whether to use deterministic actions
    """
    env = gym.make(config.ENV_ID, render_mode=None)
    
    observation, _ = env.reset()
    
    positions = []
    velocities = []
    actions = []
    rewards_per_step = []
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        positions.append(observation[0])
        velocities.append(observation[1])
        
        action, _ = model.predict(observation, deterministic=deterministic)
        actions.append(action[0])
        
        observation, reward, terminated, truncated, info = env.step(action)
        rewards_per_step.append(reward)
    
    env.close()
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Mountain Car Continuous - Episode Trajectory (Episode {episode_num})", 
                 fontsize=14, fontweight='bold')
    
    # Position over time
    axes[0, 0].plot(positions, linewidth=2, color='blue')
    axes[0, 0].axhline(y=0.45, color='green', linestyle='--', label='Goal Position')
    axes[0, 0].set_xlabel("Timestep")
    axes[0, 0].set_ylabel("Position")
    axes[0, 0].set_title("Car Position Over Time")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim([-1.3, 0.7])
    
    # Velocity over time
    axes[0, 1].plot(velocities, linewidth=2, color='orange')
    axes[0, 1].set_xlabel("Timestep")
    axes[0, 1].set_ylabel("Velocity")
    axes[0, 1].set_title("Car Velocity Over Time")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Action over time
    axes[1, 0].plot(actions, linewidth=2, color='red')
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 0].set_ylabel("Action (Force)")
    axes[1, 0].set_title("Applied Force Over Time")
    axes[1, 0].set_ylim([-1.1, 1.1])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative reward
    cumulative_rewards = np.cumsum(rewards_per_step)
    axes[1, 1].plot(cumulative_rewards, linewidth=2, color='purple')
    axes[1, 1].set_xlabel("Timestep")
    axes[1, 1].set_ylabel("Cumulative Reward")
    axes[1, 1].set_title("Cumulative Reward Over Time")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_action_space_analysis(model, num_samples=100):
    """
    Analyze the agent's policy by sampling states and plotting actions.
    
    Args:
        model: Trained RL model
        num_samples (int): Number of position samples
    """
    env = gym.make(config.ENV_ID)
    
    positions = np.linspace(-1.2, 0.6, num_samples)
    velocities = np.linspace(-0.07, 0.07, num_samples)
    
    actions = np.zeros((num_samples, num_samples))
    
    for i, pos in enumerate(positions):
        for j, vel in enumerate(velocities):
            observation = np.array([pos, vel], dtype=np.float32)
            action, _ = model.predict(observation, deterministic=True)
            actions[i, j] = action[0]
    
    env.close()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(
        actions.T,
        origin='lower',
        aspect='auto',
        cmap='RdBu_r',
        extent=[-1.2, 0.6, -0.07, 0.07]
    )
    
    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Velocity", fontsize=12)
    ax.set_title("Agent Policy: Action Values Across State Space", fontsize=14, fontweight='bold')
    
    # Add goal position line
    ax.axvline(x=0.45, color='green', linestyle='--', linewidth=2, label='Goal Position')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Action (Force)", fontsize=12)
    
    ax.legend(fontsize=10)
    plt.tight_layout()
    
    return fig


def main():
    """Main visualization entry point."""
    print("=" * 60)
    print("Mountain Car Continuous - Visualization")
    print("=" * 60)
    
    # Load model
    model_path = config.MODELS_DIR / f"{config.ALGORITHM.lower()}_mountain_car_continuous_final"
    
    if not (Path(model_path).with_suffix(".zip")).exists():
        print(f"Error: Model not found at {model_path}.zip")
        print(f"Please train a model first using train.py")
        return
    
    print(f"Loading model from: {model_path}.zip")
    algo_lower = config.ALGORITHM.lower()
    if algo_lower == "ddpg":
        model = DDPG.load(str(model_path))
    elif algo_lower == "sac":
        model = SAC.load(str(model_path))
    else:
        model = PPO.load(str(model_path))
    
    # Generate visualizations
    print("\nGenerating trajectory plot...")
    fig1 = plot_episode_trajectory(model, episode_num=1, deterministic=True)
    fig1.savefig(str(config.PROJECT_ROOT / "trajectory_visualization.png"), dpi=150, bbox_inches='tight')
    print("Saved: trajectory_visualization.png")
    
    print("\nGenerating policy analysis plot...")
    fig2 = plot_action_space_analysis(model, num_samples=50)
    fig2.savefig(str(config.PROJECT_ROOT / "policy_analysis.png"), dpi=150, bbox_inches='tight')
    print("Saved: policy_analysis.png")
    
    print("\nDisplaying plots...")
    plt.show()


if __name__ == "__main__":
    main()
