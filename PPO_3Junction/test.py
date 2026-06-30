import numpy as np
from PPO_3Junction.env import TrafficEnv3J
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

NORMALIZER_PATH = "models/vec_normalize.pkl"
MODEL_PATH      = "models/ppo_3junction"

SCENARIOS = {
    "training distribution": {
        0: [0.4, 0.4, 0.1, 0.1],
        1: [0.4, 0.4, 0.1, 0.1],
        2: [0.4, 0.4, 0.1, 0.1],
    },
    "heavy traffic": {
        0: [0.7, 0.7, 0.2, 0.2],
        1: [0.7, 0.7, 0.2, 0.2],
        2: [0.7, 0.7, 0.2, 0.2],
    },
    "light traffic": {
        0: [0.2, 0.2, 0.05, 0.05],
        1: [0.2, 0.2, 0.05, 0.05],
        2: [0.2, 0.2, 0.05, 0.05],
    },
    "N-S dominant": {
        0: [0.7, 0.7, 0.05, 0.05],
        1: [0.7, 0.7, 0.05, 0.05],
        2: [0.7, 0.7, 0.05, 0.05],
    },
    "E-W dominant": {
        0: [0.1, 0.1, 0.4, 0.4],
        1: [0.1, 0.1, 0.4, 0.4],
        2: [0.1, 0.1, 0.4, 0.4],
    },
    "unbalanced junctions": {
        0: [0.6, 0.6, 0.1, 0.1],
        1: [0.3, 0.3, 0.1, 0.1],
        2: [0.1, 0.1, 0.05, 0.05],
    },
    "rush hour J1 only": {
        0: [0.2, 0.2, 0.05, 0.05],
        1: [0.8, 0.8, 0.2, 0.2],
        2: [0.2, 0.2, 0.05, 0.05],
    },
}

def evaluate_ppo_scenario(model, rates, n_episodes=10):
    # fresh environment with exactly these rates
    raw_env = make_vec_env(
        TrafficEnv3J,
        n_envs=1,
        env_kwargs={"arrival_rates": rates}
    )
    norm_env = VecNormalize.load(NORMALIZER_PATH, raw_env)
    norm_env.training = False
    norm_env.norm_reward = False

    rewards = []
    for _ in range(n_episodes):
        obs = norm_env.reset()
        done = False
        total = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = norm_env.step(action)
            total += reward[0]
        rewards.append(total)

    norm_env.close()
    return np.mean(rewards), np.std(rewards)

def evaluate_fixed_time_scenario(rates, cycle_length=30, n_episodes=10):
    env = TrafficEnv3J(arrival_rates=rates)
    rewards = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=i)
        done = False
        total = 0
        t = 0
        while not done:
            action = [1 if (t % cycle_length == 0) else 0] * 3
            obs, reward, done, _, _ = env.step(action)
            total += reward
            t += 1
        rewards.append(total)
    return np.mean(rewards), np.std(rewards)

# load model once, reuse across all scenarios
base_env = make_vec_env(TrafficEnv3J, n_envs=1)
base_env = VecNormalize.load(NORMALIZER_PATH, base_env)
base_env.training = False
base_env.norm_reward = False
model = PPO.load(MODEL_PATH, env=base_env)

print(f"\n{'Scenario':<25} {'Fixed-time':>12} {'PPO':>12} {'Improvement':>12}")
print("─" * 65)

for name, rates in SCENARIOS.items():
    ppo_mean,   ppo_std   = evaluate_ppo_scenario(model, rates)
    fixed_mean, fixed_std = evaluate_fixed_time_scenario(rates)
    improvement = (ppo_mean - fixed_mean) / abs(fixed_mean) * 100
    print(f"{name:<25} {fixed_mean:>12.2f} {ppo_mean:>12.2f} {improvement:>11.1f}%")