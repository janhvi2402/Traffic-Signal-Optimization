import os
import numpy as np
import traci
from stable_baselines3 import PPO

from multi_env import SumoMultiJunctionEnv
from baseline import run_offset_fixed_time

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(SCRIPT_DIR, "models", "ppo_single_junction")
SUMOCFG_PATH = os.path.join(SCRIPT_DIR, "network", "multi_junction.sumocfg")

# set True to watch it in sumo-gui / record a video, False for numeric eval
RECORD = False


def run_decentralized(model, n_episodes=5):
    """
    Applies the SAME single-junction-trained model independently to J1
    and J2. The 2-junction env's 14-dim observation is [J1's 7 features,
    J2's 7 features] (see SumoMultiJunctionEnv._get_obs) — we split it,
    predict on each half separately with the shared model, then combine
    into the joint action the env.step() interface expects.
    """
    episode_waits = []

    for ep in range(n_episodes):
        env = SumoMultiJunctionEnv(
            cfg_path  = SUMOCFG_PATH,
            use_gui   = RECORD,
            max_steps = 3600,
            seed      = ep,
            port      = 8820,
        )
        obs, _ = env.reset(seed=ep)
        done = False
        steps = 0
        wait_sum = 0.0

        while not done:
            obs_j1 = obs[0:7]
            obs_j2 = obs[7:14]

            a_j1, _ = model.predict(obs_j1, deterministic=True)
            a_j2, _ = model.predict(obs_j2, deterministic=True)
            print(f"step {steps}: a_j1={int(a_j1)}, a_j2={int(a_j2)}")
            obs, reward, done, _, _ = env.step([int(a_j1), int(a_j2)])

            for veh in env.conn.vehicle.getIDList():
                wait_sum += env.conn.vehicle.getWaitingTime(veh)
            steps += 1

        episode_waits.append(wait_sum / steps)
        env.close()

    return np.mean(episode_waits), np.std(episode_waits)


def run_fixed_time(n_episodes=5):
    waits = []
    for ep in range(n_episodes):
        cmd = ["sumo-gui" if RECORD else "sumo",
               "-c", SUMOCFG_PATH, "--no-warnings", "--seed", str(ep)]
        if RECORD:
            cmd += ["--start", "--quit-on-end"]
        traci.start(cmd)
        traci.simulationStep()
        _, avg_wait, _, _ = run_offset_fixed_time(max_steps=100000)
        traci.close()
        waits.append(avg_wait)
    return np.mean(waits), np.std(waits)


# --- main ---

model = PPO.load(MODEL_PATH)

N_EP = 1 if RECORD else 5
print(f"\nEvaluating decentralized single-junction policy over {N_EP} episode(s)...\n")

decentral_wait, decentral_std = run_decentralized(model, n_episodes=N_EP)
fixed_wait, fixed_std         = run_fixed_time(n_episodes=N_EP)

improvement = (fixed_wait - decentral_wait) / fixed_wait * 100

print(f"{'Metric':<35} {'Fixed-time':>14} {'Decentralized PPO':>18}")
print("─" * 70)
print(f"{'Mean avg wait/step (s)':<35} {fixed_wait:>13.2f}s {decentral_wait:>17.2f}s")
print(f"{'Std':<35} {fixed_std:>13.2f}s {decentral_std:>17.2f}s")
print(f"\nImprovement over fixed-time: {improvement:.1f}%")
