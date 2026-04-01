"""
Demonstration: Agent Learning to Climb the Mountain using Reinforcement Learning

This script shows:
1. What the Mountain Car problem is
2. How the trained agent solves it
3. Performance metrics
"""

import gymnasium as gym
from stable_baselines3 import SAC
from pathlib import Path
import numpy as np
import config


def mountain_car_demo():
    """Demonstrate the agent climbing the mountain."""
    
    print("=" * 70)
    print("MOUNTAIN CAR CONTINUOUS - REINFORCEMENT LEARNING DEMO")
    print("=" * 70)
    print()
    
    print("📍 PROBLEM DESCRIPTION:")
    print("-" * 70)
    print("• Environment: Mountain Car Continuous")
    print("• Goal: Drive the car to the top of the right hill (position ≥ 0.45)")
    print("• Challenge: Car engine is not powerful enough to climb directly")
    print("• Solution: Rock back and forth to build momentum (swing strategy)")
    print("• State Space: [Position (-1.2 to 0.6), Velocity (-0.07 to 0.07)]")
    print("• Action Space: Continuous force [-1 (left) to +1 (right)]")
    print("• Max Steps: 999 per episode")
    print()
    
    print("🤖 LEARNING APPROACH:")
    print("-" * 70)
    print("• Algorithm: SAC (Soft Actor-Critic)")
    print("• Why SAC? Excellent exploration + stable learning")
    print("• Learns through trial and error over 200,000 timesteps")
    print("• Discovers optimal strategy: swing left-right to reach goal")
    print()
    
    # Load the trained model
    model_path = config.MODELS_DIR / "best_model"
    
    if not Path(model_path).with_suffix(".zip").exists():
        print("❌ Error: Model not found!")
        print("Please train first: python train.py")
        return
    
    print("📂 LOADING TRAINED MODEL...")
    print("-" * 70)
    model = SAC.load(str(model_path))
    print(f"✓ Model loaded: {model_path}.zip")
    print()
    
    # Test the agent
    print("🏃 TESTING TRAINED AGENT:")
    print("-" * 70)
    
    env = gym.make(config.ENV_ID)
    
    successes = 0
    total_episodes = 5
    results = []
    
    for episode in range(total_episodes):
        observation, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        # Track position over time
        positions = []
        
        while not done:
            positions.append(observation[0])  # Position
            
            # Agent predicts optimal action
            action, _ = model.predict(observation, deterministic=True)
            
            # Execute action in environment
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
        
        # Check if agent reached goal
        success = positions[-1] >= 0.45
        if success:
            successes += 1
        
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"Episode {episode + 1}: {status} | Steps: {steps:3d} | Reward: {episode_reward:7.2f}")
        results.append({
            'success': success,
            'steps': steps,
            'reward': episode_reward,
            'max_position': max(positions)
        })
    
    env.close()
    print()
    
    # Summary statistics
    print("📊 RESULTS SUMMARY:")
    print("-" * 70)
    success_rate = (successes / total_episodes) * 100
    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])
    
    print(f"Success Rate: {success_rate:.1f}% ({successes}/{total_episodes})")
    print(f"Avg Steps: {avg_steps:.1f}")
    print(f"Avg Reward: {avg_reward:.2f}")
    print()
    
    print("✅ WHAT THIS MEANS:")
    print("-" * 70)
    if success_rate >= 80:
        print("• Agent has successfully learned the optimal strategy!")
        print("• It discovered the swinging motion to build momentum")
        print("• Reaches the goal position efficiently")
        print("• Reinforcement learning worked! 🎉")
    else:
        print("• Agent is learning but needs more training")
        print("• Continue training with: python train.py")
    print()
    
    print("🎬 NEXT STEPS:")
    print("-" * 70)
    print("1. Run animation: python animate.py")
    print("2. View plots: python visualize.py")
    print("3. Full evaluation: python evaluate.py")
    print("4. See logic: python quickstart.py")
    print()
    
    print("=" * 70)


if __name__ == "__main__":
    mountain_car_demo()
