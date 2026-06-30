import numpy as np
from env import TrafficEnv2J
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

NORMALIZER_PATH = "models/vec_normalize.pkl"
MODEL_PATH      = "models/ppo_2junction"

# Arrival rate format: { junction_id: [N_IN, S_IN, E_IN, W_IN] }
#
# J1 (junction 0): W1 is external west arm → W_IN gets external rate
#                  E_IN must be 0.0 — pipeline-only (vehicles from J2)
#
# J2 (junction 1): E2 is external east arm → E_IN gets external rate
#                  W_IN must be 0.0 — pipeline-only (vehicles from J1)
#
# Format per junction: [N_IN, S_IN, E_IN, W_IN]
#                       idx0   idx1   idx2   idx3

SCENARIOS = {
    "training distribution": {
        0: [0.4, 0.4, 0.0, 0.2],   # J1: N,S heavy | W1 moderate | E pipeline only
        1: [0.4, 0.4, 0.2, 0.0],   # J2: N,S heavy | E2 moderate | W pipeline only
    },
    "heavy NS traffic": {
        0: [0.7, 0.7, 0.0, 0.2],
        1: [0.7, 0.7, 0.2, 0.0],
    },
    "light traffic": {
        0: [0.2, 0.2, 0.0, 0.05],
        1: [0.2, 0.2, 0.05, 0.0],
    },
    "N-S dominant": {
        0: [0.7, 0.7, 0.0, 0.05],
        1: [0.7, 0.7, 0.05, 0.0],
    },
    "E-W dominant": {           # heavy W1→J1→J2→E2 corridor flow
        0: [0.1, 0.1, 0.0, 0.5],
        1: [0.1, 0.1, 0.5, 0.0],
    },
    "unbalanced junctions": {   # J1 busier than J2 on NS
        0: [0.6, 0.6, 0.0, 0.15],
        1: [0.3, 0.3, 0.1, 0.0],
    },
    "rush hour J1 NS only": {   # J1 NS overwhelmed, J2 quiet
        0: [0.8, 0.8, 0.0, 0.05],
        1: [0.2, 0.2, 0.05, 0.0],
    },
    "rush hour corridor": {     # W1→E2 corridor overwhelmed, NS quiet
        0: [0.1, 0.1, 0.0, 0.7],
        1: [0.1, 0.1, 0.7, 0.0],
    },
}


def evaluate_ppo_scenario(model, rates, n_episodes=10):
    raw_env = make_vec_env(
        TrafficEnv2J,
        n_envs=1,
        env_kwargs={"arrival_rates": rates}
    )
    norm_env = VecNormalize.load(NORMALIZER_PATH, raw_env)
    norm_env.training   = False
    norm_env.norm_reward = False

    rewards = []
    for _ in range(n_episodes):
        obs  = norm_env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = norm_env.step(action)
            total += reward[0]
        rewards.append(total)

    norm_env.close()
    return np.mean(rewards), np.std(rewards)


def evaluate_fixed_time_scenario(rates, cycle_length=30, n_episodes=10):
    """
    Each junction runs its own independent fixed-time cycle.
    J2 is offset by half a cycle so they're not synchronized —
    this is a fair baseline (synchronized fixed-time would be
    artificially easy to beat).
    """
    env = TrafficEnv2J(arrival_rates=rates)
    rewards = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=i)
        done  = False
        total = 0.0
        t     = 0

        while not done:
            a0 = 1 if (t % cycle_length == 0) else 0                    # J1 cycle
            a1 = 1 if ((t + cycle_length // 2) % cycle_length == 0) else 0  # J2 offset by half
            obs, reward, done, _, _ = env.step([a0, a1])
            total += reward
            t += 1

        rewards.append(total)

    return np.mean(rewards), np.std(rewards)


# Load model once, reuse across all scenarios
base_env = make_vec_env(TrafficEnv2J, n_envs=1)
base_env = VecNormalize.load(NORMALIZER_PATH, base_env)
base_env.training   = False
base_env.norm_reward = False
model = PPO.load(MODEL_PATH, env=base_env)

print(f"\n{'Scenario':<25} {'Fixed-time':>12} {'PPO':>12} {'Improvement':>12}")
print("─" * 65)

for name, rates in SCENARIOS.items():
    ppo_mean,   ppo_std   = evaluate_ppo_scenario(model, rates)
    fixed_mean, fixed_std = evaluate_fixed_time_scenario(rates)
    improvement = (ppo_mean - fixed_mean) / abs(fixed_mean) * 100
    print(f"{name:<25} {fixed_mean:>12.2f} {ppo_mean:>12.2f} {improvement:>11.1f}%")