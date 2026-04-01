"""Quick demo: train (fast) a SAC agent and record an animation of the car climbing the mountain.

- If a saved model exists at `models/climb_model.zip`, it will be loaded.
- Otherwise the script trains a small SAC model for a short run (30k timesteps by default).
- The script then runs a few episodes with `render_mode='rgb_array'`, collects frames and
  writes `climb_animation.gif` (or MP4 if imageio-ffmpeg/ffmpeg available).

Run: python climb_demo.py
"""

from pathlib import Path
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import numpy as np
import time

MODEL_PATH = Path("models") / "climb_model"
ANIM_FILE_GIF = Path("climb_animation.gif")
ANIM_FILE_MP4 = Path("climb_animation.mp4")

# Training settings (kept small for fast demo)
TOTAL_TIMESTEPS = 30000
LEARNING_RATE = 2e-3
BATCH_SIZE = 128


def train_or_load():
    if (MODEL_PATH.with_suffix('.zip')).exists():
        print(f"Loading existing model: {MODEL_PATH}.zip")
        model = SAC.load(str(MODEL_PATH))
        return model

    print("No saved model found — training a small SAC agent (fast demo)...")
    env = gym.make("MountainCarContinuous-v0")
    env = Monitor(env)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        verbose=1,
        policy_kwargs={"net_arch": [64, 64]},
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    env.close()
    print(f"Saved model to {MODEL_PATH}.zip")
    return model


def record_animation(model, episodes=3, max_steps=999, fps=30):
    print("Running episodes and collecting frames...")
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")

    all_frames = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        frames = []
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            frame = env.render()
            if isinstance(frame, (list, tuple)):
                # Some renderers return list of frames
                frame = frame[0]
            frames.append(frame)
            steps += 1
        print(f"Episode {ep+1}: steps={steps}, final_pos={obs[0]:.3f}")
        all_frames.extend(frames)
    env.close()

    if len(all_frames) == 0:
        print("No frames captured — ensure the environment supports 'rgb_array' rendering.")
        return None

    # Try to save GIF using imageio (preferred)
    try:
        import imageio
        print("Saving animation as GIF (this may take a moment)...")
        imageio.mimsave(str(ANIM_FILE_GIF), all_frames, fps=fps)
        print(f"Saved animation: {ANIM_FILE_GIF}")
        return ANIM_FILE_GIF
    except Exception:
        print("imageio unavailable or failed — trying matplotlib+ffmpeg fallback...")

    # Fallback: try matplotlib animation writer
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps)
        fig = plt.figure(figsize=(6, 4))
        im = plt.imshow(all_frames[0])
        plt.axis('off')
        with writer.saving(fig, str(ANIM_FILE_MP4), dpi=100):
            for frame in all_frames:
                im.set_data(frame)
                writer.grab_frame()
        plt.close(fig)
        print(f"Saved animation: {ANIM_FILE_MP4}")
        return ANIM_FILE_MP4
    except Exception as e:
        print("Failed to save animation with matplotlib/ffmpeg:", e)

    # Final fallback: save individual frames as PNGs (no ffmpeg/imageio needed)
    try:
        frames_dir = Path("animation_frames")
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving {len(all_frames)} frames to: {frames_dir}")
        import matplotlib.pyplot as plt
        for idx, frame in enumerate(all_frames):
            frame_path = frames_dir / f"frame_{idx:04d}.png"
            plt.imsave(str(frame_path), frame)
        print(f"Saved frames to: {frames_dir}")
        return frames_dir
    except Exception as e2:
        print("Failed to save individual frames as PNGs:", e2)
        return None


if __name__ == '__main__':
    start = time.time()
    model = train_or_load()
    saved = record_animation(model, episodes=3, fps=25)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s. Animation saved to: {saved}")
