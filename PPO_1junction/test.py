import numpy as np
from PPO_1junction.env import TrafficEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

# Load model with the normalizer that was saved during training
env = make_vec_env(TrafficEnv, n_envs=1)
env = VecNormalize.load("models/vec_normalize.pkl", env)
env.training = False      # don't update normalizer stats during testing
env.norm_reward = False   # evaluate on raw reward so numbers are interpretable

model = PPO.load("models/ppo_traffic", env=env)

def evaluate_ppo(model, env, n_episodes=10):
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total += reward[0]   # VecEnv returns arrays, take first element
        rewards.append(total)
    return np.mean(rewards)

def evaluate_fixed_time(cycle_length=30, n_episodes=10):
    raw_env = TrafficEnv()   # fixed-time uses raw env, no normalizer needed
    rewards = []
    for _ in range(n_episodes):
        obs, _ = raw_env.reset()
        done = False
        total = 0
        t = 0
        while not done:
            action = 1 if (t % cycle_length == 0) else 0
            obs, reward, done, _, _ = raw_env.step(action)
            total += reward
            t += 1
        rewards.append(total)
    return np.mean(rewards)

ppo_score   = evaluate_ppo(model, env)
fixed_score = evaluate_fixed_time()

improvement = (ppo_score - fixed_score) / abs(fixed_score) * 100
print(f"Fixed-time baseline : {fixed_score:.2f}")
print(f"PPO agent           : {ppo_score:.2f}")
print(f"Improvement         : {improvement:.1f}%")