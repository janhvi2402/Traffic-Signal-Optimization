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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.join(SCRIPT_DIR, "..", "common")
sys.path.append(COMMON_DIR)

from baseline import run_offset_fixed_time

# Must match train.py exactly.
GREEN_TIME  = 5
YELLOW_TIME = 3

# Must match train.py exactly. GREEN_TIME=5 steps = 5s, so
# MIN_GREEN_CYCLES=2 enforces a 10-second minimum green.
MIN_GREEN_CYCLES = 2

J1 = "J1"
J2 = "J2"

# Relative actions (0=stay, 1=switch) per junction — must match train.py
# exactly, or the trained Q-table's action indices won't mean the same
# thing here.
ACTION_SPACE = [(0, 0), (0, 1), (1, 0), (1, 1)]
YELLOW_PHASE = {0: 1, 2: 3}

# --- set True when you want to record a video, False for fast headless eval ---
RECORD = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMOCFG_PATH = os.path.join(SCRIPT_DIR, "simulation.sumocfg")
QTABLE_PATH  = os.path.join(SCRIPT_DIR, "qtable.pkl")

print(f"[DEBUG] Script dir      : {SCRIPT_DIR}")
print(f"[DEBUG] Looking for cfg : {SUMOCFG_PATH}")
print(f"[DEBUG] cfg exists?     : {os.path.exists(SUMOCFG_PATH)}")
print(f"[DEBUG] Looking for qtb : {QTABLE_PATH}")
print(f"[DEBUG] qtable exists?  : {os.path.exists(QTABLE_PATH)}")

if not os.path.exists(SUMOCFG_PATH):
    sys.exit(
        f"\nFATAL: simulation.sumocfg not found at:\n  {SUMOCFG_PATH}\n"
        f"Contents of {SCRIPT_DIR}:\n  " +
        "\n  ".join(sorted(os.listdir(SCRIPT_DIR)))
    )
if not os.path.exists(QTABLE_PATH):
    sys.exit(f"\nFATAL: qtable.pkl not found at:\n  {QTABLE_PATH}")

with open(QTABLE_PATH, "rb") as f:
    q_table = pickle.load(f)

print(f"States loaded: {len(q_table)}")


def bucket(x):
    """
    MUST stay identical to bucket() in train.py — this is the function
    that defines the Q-table's state keys. If this drifts from train.py,
    lookups here will silently miss and fall back to action 0 every time.
    """
    if x == 0:     return 0
    elif x <= 2:   return 1
    elif x <= 5:   return 2
    elif x <= 8:   return 3
    elif x <= 11:  return 4
    elif x <= 15:  return 5
    elif x <= 20:  return 6
    else:          return 7


def bucket_hold(cycles):
    """
    CHANGED: was 3 values, now 4 — separates "forced to stay" (hold below
    the min-green floor, switch is masked to a no-op) from "floor just
    reached" (first cycle where switching actually does something).
    MUST stay identical to bucket_hold() in train.py.
    """
    if cycles < MIN_GREEN_CYCLES:         return 0
    elif cycles == MIN_GREEN_CYCLES:      return 1
    elif cycles <= MIN_GREEN_CYCLES + 2:  return 2
    else:                                  return 3


def get_halted(lane_id):
    return traci.lane.getLastStepHaltingNumber(lane_id)


def get_arm_queues(which, phase):
    """Raw halted counts for the currently-green / currently-red arm of a
    junction — must match train.py's version."""
    if which == "J1":
        if phase == 0:
            green = get_halted("N1_J1_0") + get_halted("S1_J1_0")
            red   = get_halted("W_J1_0")  + get_halted("J2_J1_0")
        else:
            green = get_halted("W_J1_0")  + get_halted("J2_J1_0")
            red   = get_halted("N1_J1_0") + get_halted("S1_J1_0")
    else:
        if phase == 0:
            green = get_halted("N2_J2_0") + get_halted("S2_J2_0")
            red   = get_halted("J1_J2_0") + get_halted("E_J2_0")
        else:
            green = get_halted("J1_J2_0") + get_halted("E_J2_0")
            red   = get_halted("N2_J2_0") + get_halted("S2_J2_0")
    return green, red


def get_state(j1_phase, j2_phase, j1_hold, j2_hold):
    j1_green, j1_red = get_arm_queues("J1", j1_phase)
    j2_green, j2_red = get_arm_queues("J2", j2_phase)

    return (
        bucket(j1_green), bucket(j1_red),
        bucket(j2_green), bucket(j2_red),
        j1_phase // 2,
        j2_phase // 2,
        bucket_hold(j1_hold),
        bucket_hold(j2_hold),
    )


def _sumo_cmd(seed):
    cmd = ["sumo-gui" if RECORD else "sumo",
           "-c", SUMOCFG_PATH, "--no-warnings", "--seed", str(seed)]
    if RECORD:
        cmd += ["--start", "--quit-on-end"]
    return cmd


def run_qlearning_episode(seed):
    traci.start(_sumo_cmd(seed))
    traci.simulationStep()

    j1_phase = 0
    j2_phase = 0
    j1_hold  = 0
    j2_hold  = 0
    traci.trafficlight.setPhase(J1, j1_phase)
    traci.trafficlight.setPhaseDuration(J1, 9999)
    traci.trafficlight.setPhase(J2, j2_phase)
    traci.trafficlight.setPhaseDuration(J2, 9999)

    cumulative_wait = 0.0
    arrived = 0
    sim_steps = 0
    unseen_states = 0
    total_decisions = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        state = get_state(j1_phase, j2_phase, j1_hold, j2_hold)
        total_decisions += 1
        if state in q_table:
            action_idx = int(np.argmax(q_table[state]))
        else:
            unseen_states += 1
            action_idx = 0

        j1_choice, j2_choice = ACTION_SPACE[action_idx]
        # Same MIN_GREEN_CYCLES mask used during training — a chosen
        # "switch" only actually happens if the junction has held long enough.
        j1_switching = bool(j1_choice) and (j1_hold >= MIN_GREEN_CYCLES)
        j2_switching = bool(j2_choice) and (j2_hold >= MIN_GREEN_CYCLES)

        j1_new = (2 - j1_phase) if j1_switching else j1_phase
        j2_new = (2 - j2_phase) if j2_switching else j2_phase

        if j1_switching:
            traci.trafficlight.setPhase(J1, YELLOW_PHASE[j1_phase])
            traci.trafficlight.setPhaseDuration(J1, 9999)
        if j2_switching:
            traci.trafficlight.setPhase(J2, YELLOW_PHASE[j2_phase])
            traci.trafficlight.setPhaseDuration(J2, 9999)

        if j1_switching or j2_switching:
            for _ in range(YELLOW_TIME):
                traci.simulationStep()
                sim_steps += 1
                arrived += len(traci.simulation.getArrivedIDList())
                for veh in traci.vehicle.getIDList():
                    cumulative_wait += traci.vehicle.getWaitingTime(veh)

        traci.trafficlight.setPhase(J1, j1_new)
        traci.trafficlight.setPhaseDuration(J1, 9999)
        traci.trafficlight.setPhase(J2, j2_new)
        traci.trafficlight.setPhaseDuration(J2, 9999)
        j1_phase, j2_phase = j1_new, j2_new
        j1_hold = 0 if j1_switching else j1_hold + 1
        j2_hold = 0 if j2_switching else j2_hold + 1

        for _ in range(GREEN_TIME):
            traci.simulationStep()
            sim_steps += 1
            arrived += len(traci.simulation.getArrivedIDList())
            for veh in traci.vehicle.getIDList():
                cumulative_wait += traci.vehicle.getWaitingTime(veh)

    traci.close()
    coverage = 1.0 - (unseen_states / max(total_decisions, 1))
    return cumulative_wait, cumulative_wait / max(sim_steps, 1), arrived, sim_steps, coverage


N_EPISODES = 1 if RECORD else 5
ql_waits = []
fixed_waits = []
coverages = []
ql_arrived = []
ql_steps_list = []

for ep in range(N_EPISODES):
    # Fixed-time baseline
    traci.start(_sumo_cmd(ep))
    traci.simulationStep()
    _, fixed_avg_wait, _, fixed_steps = run_offset_fixed_time(max_steps=100000)
    print(f"[DEBUG] Episode {ep}: fixed-time ran {fixed_steps} steps before clearing")
    traci.close()
    fixed_waits.append(fixed_avg_wait)

    # Q-learning
    _, ql_avg_wait, arrived, steps, coverage = run_qlearning_episode(seed=ep)
    ql_waits.append(ql_avg_wait)
    coverages.append(coverage)
    ql_arrived.append(arrived)
    ql_steps_list.append(steps)

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
        "fixed_waits_per_episode": [float(x) for x in fixed_waits],
        "ql_waits_per_episode": [float(x) for x in ql_waits],
        "coverage_per_episode": [float(x) for x in coverages],
        "ql_arrived_per_episode": [int(x) for x in ql_arrived],
        "ql_steps_per_episode": [int(x) for x in ql_steps_list],
    }, f, indent=2)

print(f"Saved: {os.path.join(RESULTS_DIR, 'qlearning_vs_fixed_unified.json')}")