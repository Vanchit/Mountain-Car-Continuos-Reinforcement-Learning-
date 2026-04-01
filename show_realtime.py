"""Show MountainCarContinuous agent in real-time using the system display.

This script loads `models/climb_model.zip` if present, otherwise trains a small SAC
agent (fast demo) and then runs a few episodes with `render_mode='human'`.

Run:
    python show_realtime.py
"""

from pathlib import Path
import time
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

MODEL_PATH = Path("models") / "climb_model"


def train_or_load():
    if (MODEL_PATH.with_suffix('.zip')).exists():
        print(f"Loading existing model: {MODEL_PATH}.zip")
        return SAC.load(str(MODEL_PATH))

    print("No saved model found — training a small SAC agent (fast demo)...")
    env = gym.make("MountainCarContinuous-v0")
    env = Monitor(env)
    model = SAC("MlpPolicy", env, learning_rate=2e-3, batch_size=128, verbose=1,
                policy_kwargs={"net_arch": [64, 64]})
    model.learn(total_timesteps=30000)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    env.close()
    print(f"Saved model to {MODEL_PATH}.zip")
    return model


def run_realtime(model, episodes=3, max_steps=999, pause=0.02):
    print("Starting real-time rendering. A window should open on your desktop.")
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done and steps < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                # env.render() is handled by the environment in human mode
                time.sleep(pause)
                steps += 1
            print(f"Episode {ep+1} finished after {steps} steps, final_pos={obs[0]:.3f}")
    finally:
        env.close()


if __name__ == '__main__':
    model = train_or_load()
    run_realtime(model, episodes=3, pause=0.02)
