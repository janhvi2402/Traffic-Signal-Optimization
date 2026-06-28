import numpy as np
from env import TrafficEnv3J
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

def evaluate_ppo(model, env, n_episodes=10):
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total += reward[0]
        rewards.append(total)
    return np.mean(rewards)

def evaluate_fixed_time(cycle_length=30, n_episodes=10):
    raw_env = TrafficEnv3J()
    rewards = []
    for i in range(n_episodes):
        obs, _ = raw_env.reset(seed=i)
        done = False
        total = 0
        t = 0
        while not done:
            # same fixed cycle for all 3 junctions
            action = [1 if (t % cycle_length == 0) else 0] * 3
            obs, reward, done, _, _ = raw_env.step(action)
            total += reward
            t += 1
        rewards.append(total)
    return np.mean(rewards)

env = make_vec_env(TrafficEnv3J, n_envs=1)
env = VecNormalize.load("models/vec_normalize.pkl", env)
env.training = False
env.norm_reward = False

model = PPO.load("models/ppo_3junction", env=env)

ppo_score   = evaluate_ppo(model, env)
fixed_score = evaluate_fixed_time()

improvement = (ppo_score - fixed_score) / abs(fixed_score) * 100
print(f"Fixed-time baseline : {fixed_score:.2f}")
print(f"PPO agent           : {ppo_score:.2f}")
print(f"Improvement         : {improvement:.1f}%")