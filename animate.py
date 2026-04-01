"""Animation script to visualize the trained agent driving the Mountain Car."""

import gymnasium as gym
from stable_baselines3 import SAC
from pathlib import Path
import config


def animate_agent(model_path=None, episodes=1, render=True):
    """
    Animate the agent driving the Mountain Car.
    
    Args:
        model_path (str): Path to the trained model
        episodes (int): Number of episodes to animate
        render (bool): Whether to render with GUI (True) or just stats (False)
    """
    # Load model
    if model_path is None:
        model_path = config.MODELS_DIR / "best_model"
    
    if not Path(model_path).with_suffix(".zip").exists():
        print(f"Error: Model not found at {model_path}.zip")
        print("Please train a model first using: python train.py")
        return
    
    print(f"Loading model from: {model_path}.zip")
    model = SAC.load(str(model_path))
    
    # Create environment
    render_mode = "human" if render else None
    env = gym.make(config.ENV_ID, render_mode=render_mode)
    
    print(f"\nRunning {episodes} episode(s) with animation...")
    if render:
        print("(Animation window will open - close it to continue)\n")
    else:
        print()
    
    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        observation, info = env.reset()
        done = False
        score = 0
        steps = 0
        
        while not done:
            # Get action from model
            action, _ = model.predict(observation, deterministic=True)
            
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            score += reward
            steps += 1
        
        print(f"  ✓ Score: {score:.2f}, Steps: {steps}")
    
    env.close()
    print("\n✓ Animation complete!")


if __name__ == "__main__":
    animate_agent(episodes=3)
