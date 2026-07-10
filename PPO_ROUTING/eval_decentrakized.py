import os
import numpy as np
import traci
from stable_baselines3 import PPO

from multi_env import SumoMultiJunctionEnv
from baseline import run_offset_fixed_time

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(SCRIPT_DIR, "models", "ppo_single_junction")
SUMOCFG_PATH = os.path.join(SCRIPT_DIR, "network", "multi_junction.sumocfg")

RECORD = False


def run_decentralized(model, n_episodes=5):
    """
    CHANGED: obs is now 16-dim (8 features x 2 junctions), was 14-dim.
    Split points updated to obs[0:8] / obs[8:16] to match the retrained
    model's SumoSingleJunctionEnv observation layout.
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
            obs_j1 = obs[0:8]
            obs_j2 = obs[8:16]

            a_j1, _ = model.predict(obs_j1, deterministic=True)
            a_j2, _ = model.predict(obs_j2, deterministic=True)

            if steps % 50 == 0:
                print(
                    f"step {steps}: "
                    f"q_j1={obs_j1[0]:.2f}/{obs_j1[1]:.2f} imb_j1={obs_j1[7]:.2f} a_j1={int(a_j1)} | "
                    f"q_j2={obs_j2[0]:.2f}/{obs_j2[1]:.2f} imb_j2={obs_j2[7]:.2f} a_j2={int(a_j2)}"
                )

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


model = PPO.load(MODEL_PATH)

N_EP = 1 if RECORD else 5
print(f"\nEvaluating decentralized single-junction policy over {N_EP} episode(s)...\n")

decentral_wait, decentral_std = run_decentralized(model, n_episodes=N_EP)
fixed_wait, fixed_std         = run_fixed_time(n_episodes=N_EP)

improvement = (fixed_wait - decentral_wait) / fixed_wait * 100

print(f"{'Metric':<35} {'Fixed-time':>14} {'Decentralized PPO':>18}")
print("-" * 70)
print(f"{'Mean avg wait/step (s)':<35} {fixed_wait:>13.2f}s {decentral_wait:>17.2f}s")
print(f"{'Std':<35} {fixed_std:>13.2f}s {decentral_std:>17.2f}s")
print(f"\nImprovement over fixed-time: {improvement:.1f}%")

# Diagnostic: are J1 and J2 actually acting differently when their
# queues differ? Quick correlation check across one episode.
print("\n--- Divergence check ---")
env = SumoMultiJunctionEnv(cfg_path=SUMOCFG_PATH, use_gui=False, max_steps=3600, seed=99, port=8821)
obs, _ = env.reset(seed=99)
actions_j1, actions_j2, queue_diffs = [], [], []
done = False
while not done:
    obs_j1, obs_j2 = obs[0:8], obs[8:16]
    a_j1, _ = model.predict(obs_j1, deterministic=True)
    a_j2, _ = model.predict(obs_j2, deterministic=True)
    actions_j1.append(int(a_j1))
    actions_j2.append(int(a_j2))
    queue_diffs.append(abs(obs_j1[0] - obs_j2[0]) + abs(obs_j1[1] - obs_j2[1]))
    obs, _, done, _, _ = env.step([int(a_j1), int(a_j2)])
env.close()

same_action_rate = np.mean(np.array(actions_j1) == np.array(actions_j2))
print(f"J1/J2 same-action rate: {same_action_rate:.1%}  (lower is better -- "
      f"near 100% means it's still not conditioning on local queue state)")