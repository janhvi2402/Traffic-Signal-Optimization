import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from baseline import run_offset_fixed_time

from env import SumoTrafficEnv2J

MODEL_PATH      = "models/ppo_sumo_2junction"
NORMALIZER_PATH = "models/vec_normalize_sumo.pkl"

#  helpers 

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

    episode_waits = []

    for ep in range(n_episodes):
        obs   = env.reset()
        done  = False
        steps = 0
        wait_sum = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            conn = env.get_attr("conn")[0]
            for veh in conn.vehicle.getIDList():
                wait_sum += conn.vehicle.getWaitingTime(veh)
            steps += 1

        episode_waits.append(wait_sum / steps)

    env.close()
    return np.mean(episode_waits), np.std(episode_waits)


def run_fixed_time(n_episodes=5):
    waits = []
    for ep in range(n_episodes):
        traci.start(["sumo", "-c", "network.sumocfg", "--no-warnings", "--seed", str(ep)])
        traci.simulationStep()
        _, avg_wait, _, _ = run_offset_fixed_time()
        traci.close()
        waits.append(avg_wait)
    return np.mean(waits), np.std(waits)

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


# main 

base_env = make_vec_env(make_env(seed=0), n_envs=1)
base_env = VecNormalize.load(NORMALIZER_PATH, base_env)
base_env.training    = False
base_env.norm_reward = False
model = PPO.load(MODEL_PATH, env=base_env)

N_EP = 5
print(f"\nEvaluating over {N_EP} episodes each...\n")

ppo_wait, ppo_wait_std     = run_ppo(model, n_episodes=N_EP)
fixed_wait, fixed_wait_std = run_fixed_time(n_episodes=N_EP)

improvement = (fixed_wait - ppo_wait) / fixed_wait * 100

print(f"{'Metric':<30} {'Fixed-time':>14} {'PPO':>14}")
print("─" * 60)
print(f"{'Mean avg wait/step (s)':<30} {fixed_wait:>13.2f}s {ppo_wait:>13.2f}s")
print(f"{'Std':<30} {fixed_wait_std:>13.2f}s {ppo_wait_std:>13.2f}s")
print(f"\nImprovement over fixed-time: {improvement:.1f}%")