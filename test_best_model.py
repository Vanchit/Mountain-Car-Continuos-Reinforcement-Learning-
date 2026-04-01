"""Quick test of the best model."""
from stable_baselines3 import DDPG
import gymnasium as gym

model = DDPG.load('models/best_model')
env = gym.make('MountainCarContinuous-v0')

successes = 0
for ep in range(5):
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 999:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        steps += 1
        done = term or trunc
    
    if term:
        print(f'Episode {ep+1}: SUCCESS in {steps} steps!')
        successes += 1
    else:
        print(f'Episode {ep+1}: Timeout at {steps} steps')

print(f'\nSuccess Rate: {successes}/5 ({100*successes/5:.0f}%)')
env.close()
