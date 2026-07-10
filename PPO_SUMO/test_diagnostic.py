import os
import sys
import numpy as np
import traci
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from baseline import run_offset_fixed_time

from env import SumoTrafficEnv2J

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(SCRIPT_DIR, "models", "ppo_sumo_2junction")
NORMALIZER_PATH = os.path.join(SCRIPT_DIR, "models", "vec_normalize_sumo.pkl")

# --- set True when you want to record a video, False for fast headless eval ---
RECORD = True

# --- helpers ---

def make_env(seed=0):
    def _init():
        return SumoTrafficEnv2J(
            cfg_path  = os.path.join(SCRIPT_DIR, "network.sumocfg"),
            use_gui   = RECORD,
            max_steps = 3600,
            seed      = seed,
        )
    return _init


def run_ppo(model, n_episodes=5):
    episode_waits = []

    for ep in range(n_episodes):
        # FIX: build a fresh env with a NEW seed each episode. Previously
        # this was built once outside the loop with seed=99 hardcoded, so
        # every "episode" ran the identical simulation — that's why std
        # came out as exactly 0.00. Each episode now gets its own seed,
        # matching the pattern already used by run_fixed_time().
        raw = make_vec_env(make_env(seed=ep), n_envs=1)
        env = VecNormalize.load(NORMALIZER_PATH, raw)
        env.training    = False
        env.norm_reward = False

        obs   = env.reset()
        done  = False
        steps = 0
        wait_sum = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # DIAGNOSTIC: is the centralized policy actually reacting to
            # queue asymmetry between J1/J2, or just moving them together?
            # obs is normalized: [0:7]=J1 features, [7:14]=J2 features
            # (same layout as the single-junction obs, see env.py _get_obs)
            flat_obs = obs[0] if obs.ndim > 1 else obs   # VecEnv wraps in a batch dim
            q_j1 = flat_obs[0]   # J1 NS-queue feature
            q_j2 = flat_obs[7]   # J2 NS-queue feature
            a_j1, a_j2 = int(action[0][0]), int(action[0][1])
            print(f"step {steps}: q_j1={q_j1:.2f} a_j1={a_j1} | q_j2={q_j2:.2f} a_j2={a_j2}")

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
        cmd = ["sumo-gui" if RECORD else "sumo",
               "-c", os.path.join(SCRIPT_DIR, "network.sumocfg"),
               "--no-warnings", "--seed", str(ep)]
        if RECORD:
            cmd += ["--start", "--quit-on-end"]   # auto-play + auto-close so the script can continue
        traci.start(cmd)
        traci.simulationStep()
        _, avg_wait, _, _ = run_offset_fixed_time(max_steps=100000)
        traci.close()
        waits.append(avg_wait)
    return np.mean(waits), np.std(waits)


# --- main ---

base_env = make_vec_env(make_env(seed=0), n_envs=1)
base_env = VecNormalize.load(NORMALIZER_PATH, base_env)
base_env.training    = False
base_env.norm_reward = False
model = PPO.load(MODEL_PATH, env=base_env)

# When recording, do 1 episode — GUI runs in real time, 5 episodes will take a while
N_EP = 1 if RECORD else 5
print(f"\nEvaluating over {N_EP} episode(s) each...\n")

ppo_wait, ppo_wait_std     = run_ppo(model, n_episodes=N_EP)
fixed_wait, fixed_wait_std = run_fixed_time(n_episodes=N_EP)

improvement = (fixed_wait - ppo_wait) / fixed_wait * 100

print(f"{'Metric':<30} {'Fixed-time':>14} {'PPO':>14}")
print("─" * 60)
print(f"{'Mean avg wait/step (s)':<30} {fixed_wait:>13.2f}s {ppo_wait:>13.2f}s")
print(f"{'Std':<30} {fixed_wait_std:>13.2f}s {ppo_wait_std:>13.2f}s")
print(f"\nImprovement over fixed-time: {improvement:.1f}%")