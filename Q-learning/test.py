import os
import sys
import numpy as np
import pickle
import json

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

import traci
from route_generator import generate_route_file
from common_baseline import run_offset_fixed_time  # SAME baseline used for PPO comparison

GREEN_TIME  = 10
YELLOW_TIME = 3

J1 = "J1"
J2 = "J2"

ACTION_SPACE = [(0, 0), (0, 2), (2, 0), (2, 2)]
YELLOW_PHASE = {0: 1, 2: 3}

with open("qtable.pkl", "rb") as f:
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


def run_qlearning_episode():
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
            action_idx = 0   # fallback: keep/default to NS-NS green

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

    coverage = 1.0 - (unseen_states / max(total_decisions, 1))
    return cumulative_wait, cumulative_wait / max(sim_steps, 1), arrived, sim_steps, coverage


SCENARIOS = ["low", "medium", "high", "asymmetric", "rush_hour"]

results = {}
print(f"\n{'Scenario':<15} {'Fixed-time':>14} {'Q-learning':>14} {'Improvement':>13} {'State cov.':>11}")
print("─" * 72)

for scenario in SCENARIOS:
    generate_route_file(scenario)

    # Fixed-time baseline — SAME offset logic used in PPO's evaluation
    traci.start(["sumo", "-c", "test.sumocfg", "--no-warnings"])
    traci.simulationStep()
    _, fixed_avg_wait, _, _ = run_offset_fixed_time()
    traci.close()

    # Q-learning
    generate_route_file(scenario)   # regenerate — baseline run consumed the route file's vehicles
    traci.start(["sumo", "-c", "test.sumocfg", "--no-warnings"])
    traci.simulationStep()
    _, ql_avg_wait, arrived, steps, coverage = run_qlearning_episode()
    traci.close()

    improvement = (fixed_avg_wait - ql_avg_wait) / fixed_avg_wait * 100

    results[scenario] = {
        "fixed_avg_wait": fixed_avg_wait,
        "qlearning_avg_wait": ql_avg_wait,
        "improvement_pct": improvement,
        "state_coverage": coverage,
        "vehicles_arrived": arrived,
    }

    print(f"{scenario:<15} {fixed_avg_wait:>13.2f}s {ql_avg_wait:>13.2f}s {improvement:>12.1f}% {coverage:>10.1%}")

os.makedirs("results", exist_ok=True)
with open("results/qlearning_vs_fixed_unified.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved: results/qlearning_vs_fixed_unified.json")
print("\nNote: 'State coverage' shows the fraction of decisions where the agent")
print("recognized the state from training. Low coverage on a scenario means")
print("results there are less trustworthy — the agent was mostly guessing/defaulting.")