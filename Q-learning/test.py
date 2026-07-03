import os
import sys
import json
import numpy as np
import pickle

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

import traci
from common_baseline import run_offset_fixed_time

GREEN_TIME  = 10
YELLOW_TIME = 3

J1 = "J1"
J2 = "J2"

ACTION_SPACE = [(0, 0), (0, 2), (2, 0), (2, 2)]
YELLOW_PHASE = {0: 1, 2: 3}

# --- FIX: resolve paths relative to this script's own folder, not the cwd ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMOCFG_PATH = os.path.join(SCRIPT_DIR, "simulation.sumocfg")
QTABLE_PATH  = os.path.join(SCRIPT_DIR, "qtable.pkl")

# --- DIAGNOSTIC: fail loudly, with the exact path, instead of a vague SUMO error ---
print(f"[DEBUG] Script dir      : {SCRIPT_DIR}")
print(f"[DEBUG] Looking for cfg : {SUMOCFG_PATH}")
print(f"[DEBUG] cfg exists?     : {os.path.exists(SUMOCFG_PATH)}")
print(f"[DEBUG] Looking for qtb : {QTABLE_PATH}")
print(f"[DEBUG] qtable exists?  : {os.path.exists(QTABLE_PATH)}")

if not os.path.exists(SUMOCFG_PATH):
    sys.exit(
        f"\nFATAL: test.sumocfg not found at:\n  {SUMOCFG_PATH}\n"
        f"Contents of {SCRIPT_DIR}:\n  " +
        "\n  ".join(sorted(os.listdir(SCRIPT_DIR)))
    )
if not os.path.exists(QTABLE_PATH):
    sys.exit(f"\nFATAL: qtable.pkl not found at:\n  {QTABLE_PATH}")

with open(QTABLE_PATH, "rb") as f:
    q_table = pickle.load(f)

print(f"States loaded: {len(q_table)}")


def bucket(x):
    if x == 0:    return 0
    elif x <= 2:  return 1
    elif x <= 6:  return 2
    elif x <= 12: return 3
    else:         return 4


def get_halted(lane_id):
    return traci.lane.getLastStepHaltingNumber(lane_id)


def get_state(j1_phase, j2_phase):
    if j1_phase == 0:
        j1_green = get_halted("N1_J1_0") + get_halted("S1_J1_0")
        j1_red   = get_halted("W_J1_0")  + get_halted("J2_J1_0")
    else:
        j1_green = get_halted("W_J1_0")  + get_halted("J2_J1_0")
        j1_red   = get_halted("N1_J1_0") + get_halted("S1_J1_0")

    if j2_phase == 0:
        j2_green = get_halted("N2_J2_0") + get_halted("S2_J2_0")
        j2_red   = get_halted("J1_J2_0") + get_halted("E_J2_0")
    else:
        j2_green = get_halted("J1_J2_0") + get_halted("E_J2_0")
        j2_red   = get_halted("N2_J2_0") + get_halted("S2_J2_0")

    return (
        bucket(j1_green), bucket(j1_red),
        bucket(j2_green), bucket(j2_red),
        j1_phase // 2,
        j2_phase // 2,
    )


def run_qlearning_episode(seed):
    traci.start(["sumo", "-c", SUMOCFG_PATH, "--no-warnings", "--seed", str(seed)])
    traci.simulationStep()

    j1_phase = 0
    j2_phase = 0
    traci.trafficlight.setPhase(J1, j1_phase)
    traci.trafficlight.setPhase(J2, j2_phase)

    cumulative_wait = 0.0
    arrived = 0
    sim_steps = 0
    unseen_states = 0
    total_decisions = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        state = get_state(j1_phase, j2_phase)
        total_decisions += 1
        if state in q_table:
            action_idx = int(np.argmax(q_table[state]))
        else:
            unseen_states += 1
            action_idx = 0

        j1_new, j2_new = ACTION_SPACE[action_idx]

        if j1_new != j1_phase:
            traci.trafficlight.setPhase(J1, YELLOW_PHASE[j1_phase])
        if j2_new != j2_phase:
            traci.trafficlight.setPhase(J2, YELLOW_PHASE[j2_phase])

        if j1_new != j1_phase or j2_new != j2_phase:
            for _ in range(YELLOW_TIME):
                traci.simulationStep()
                sim_steps += 1
                arrived += len(traci.simulation.getArrivedIDList())
                for veh in traci.vehicle.getIDList():
                    cumulative_wait += traci.vehicle.getWaitingTime(veh)

        traci.trafficlight.setPhase(J1, j1_new)
        traci.trafficlight.setPhase(J2, j2_new)
        j1_phase, j2_phase = j1_new, j2_new

        for _ in range(GREEN_TIME):
            traci.simulationStep()
            sim_steps += 1
            arrived += len(traci.simulation.getArrivedIDList())
            for veh in traci.vehicle.getIDList():
                cumulative_wait += traci.vehicle.getWaitingTime(veh)

    traci.close()
    coverage = 1.0 - (unseen_states / max(total_decisions, 1))
    return cumulative_wait, cumulative_wait / max(sim_steps, 1), arrived, sim_steps, coverage


N_EPISODES = 5
ql_waits = []
fixed_waits = []
coverages = []

for ep in range(N_EPISODES):
    # Fixed-time baseline
    traci.start(["sumo", "-c", SUMOCFG_PATH, "--no-warnings", "--seed", str(ep)])
    traci.simulationStep()
    _, fixed_avg_wait, _, _ = run_offset_fixed_time()
    traci.close()
    fixed_waits.append(fixed_avg_wait)

    # Q-learning
    _, ql_avg_wait, arrived, steps, coverage = run_qlearning_episode(seed=ep)
    ql_waits.append(ql_avg_wait)
    coverages.append(coverage)

fixed_mean = np.mean(fixed_waits)
ql_mean    = np.mean(ql_waits)
improvement = (fixed_mean - ql_mean) / fixed_mean * 100

print(f"\n{'Metric':<25} {'Fixed-time':>14} {'Q-learning':>14}")
print("─" * 55)
print(f"{'Mean avg wait/step (s)':<25} {fixed_mean:>13.2f}s {ql_mean:>13.2f}s")
print(f"{'Std':<25} {np.std(fixed_waits):>13.2f}s {np.std(ql_waits):>13.2f}s")
print(f"\nImprovement over fixed-time: {improvement:.1f}%")
print(f"Mean state coverage: {np.mean(coverages):.1%}")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, "qlearning_vs_fixed_unified.json"), "w") as f:
    json.dump({
        "fixed_mean_wait": fixed_mean,
        "qlearning_mean_wait": ql_mean,
        "improvement_pct": improvement,
        "mean_state_coverage": float(np.mean(coverages)),
    }, f, indent=2)

print(f"Saved: {os.path.join(RESULTS_DIR, 'qlearning_vs_fixed_unified.json')}")