import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from env import SumoTrafficEnv2J

MODEL_PATH      = "models/ppo_sumo_2junction"
NORMALIZER_PATH = "models/vec_normalize_sumo.pkl"

# ── helpers ───────────────────────────────────────────────────────────────────

def make_env(seed=0):
    def _init():
        return SumoTrafficEnv2J(
            cfg_path  = os.path.join(os.path.dirname(__file__), "network.sumocfg"),
            use_gui   = False,
            max_steps = 3600,
            seed      = seed,
        )
    return _init


def run_ppo(model, n_episodes=5):
    raw = make_vec_env(make_env(seed=99), n_envs=1)
    env = VecNormalize.load(NORMALIZER_PATH, raw)
    env.training    = False
    env.norm_reward = False

    episode_rewards = []
    episode_queues  = []

    for ep in range(n_episodes):
        obs   = env.reset()
        done  = False
        total = 0.0
        steps = 0
        queue_sum = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total     += reward[0]
            queue_sum += -reward[0] * env.unwrapped.MAX_QUEUE   # unnormalise
            steps     += 1

        episode_rewards.append(total)
        episode_queues.append(queue_sum / steps)

    env.close()
    return np.mean(episode_rewards), np.std(episode_rewards), np.mean(episode_queues)


def run_fixed_time(cycle_ns=42, cycle_ew=42, yellow=3, n_episodes=5):
    """
    Mimics the static tlLogic from the net file:
      42 s NS green → 3 s yellow → 42 s EW green → 3 s yellow → repeat
    J2 starts offset by half a full cycle so they're not synchronised.
    """
    full_cycle = cycle_ns + yellow + cycle_ew + yellow   # 90 s
    half_cycle = full_cycle // 2                          # 45 s

    env = SumoTrafficEnv2J(
        cfg_path  = os.path.join(os.path.dirname(__file__), "network.sumocfg"),
        use_gui   = False,
        max_steps = 3600,
    )

    episode_rewards = []
    episode_queues  = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done   = False
        total  = 0.0
        steps  = 0
        queue_sum = 0.0
        t = 0

        while not done:
            # J1: phase position in cycle
            pos_j1 = t % full_cycle
            # J2: offset by half cycle
            pos_j2 = (t + half_cycle) % full_cycle

            def want_switch(pos):
                # switch at the boundary between green phases
                return 1 if pos in (cycle_ns, cycle_ns + yellow + cycle_ew) else 0

            action = [want_switch(pos_j1), want_switch(pos_j2)]
            obs, reward, done, _, _ = env.step(action)
            total     += reward
            queue_sum += -reward * env.MAX_QUEUE
            steps     += 1
            t         += 1

        episode_rewards.append(total)
        episode_queues.append(queue_sum / steps)

    env.close()
    return np.mean(episode_rewards), np.std(episode_rewards), np.mean(episode_queues)


# ── main ──────────────────────────────────────────────────────────────────────

base_env = make_vec_env(make_env(seed=0), n_envs=1)
base_env = VecNormalize.load(NORMALIZER_PATH, base_env)
base_env.training    = False
base_env.norm_reward = False
model = PPO.load(MODEL_PATH, env=base_env)

N_EP = 5
print(f"\nEvaluating over {N_EP} episodes each...\n")

ppo_r,   ppo_std,   ppo_q   = run_ppo(model,        n_episodes=N_EP)
fixed_r, fixed_std, fixed_q = run_fixed_time(        n_episodes=N_EP)

improvement = (ppo_r - fixed_r) / abs(fixed_r) * 100

print(f"{'Metric':<30} {'Fixed-time':>12} {'PPO':>12}")
print("─" * 56)
print(f"{'Mean episode reward':<30} {fixed_r:>12.2f} {ppo_r:>12.2f}")
print(f"{'Reward std':<30} {fixed_std:>12.2f} {ppo_std:>12.2f}")
print(f"{'Mean queue (vehicles/lane)':<30} {fixed_q:>12.2f} {ppo_q:>12.2f}")
print(f"\nImprovement over fixed-time: {improvement:.1f}%")