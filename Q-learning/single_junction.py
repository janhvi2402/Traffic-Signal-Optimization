import os
import sys
import numpy as np
import pickle

# SUMO setup
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

import traci


# ---------------- LOAD TRAINED MODEL ----------------
with open("qtable.pkl", "rb") as f:
    q_table = pickle.load(f)


# ---------------- CONFIG ----------------
J1 = "J1"

GREEN_TIME = 6
YELLOW_TIME = 3

ACTION_SPACE = [
    0,  # phase A
    2   # phase B
]


# ---------------- STATE ----------------
def bucket(x):
    if x == 0:
        return 0
    elif x <= 2:
        return 1
    elif x <= 4:
        return 2
    elif x <= 6:
        return 3
    elif x <= 8:
        return 4
    elif x <= 10:
        return 5
    else:
        return 6


def lane_count(edge):
    return traci.lane.getLastStepVehicleNumber(edge + "_0")


def get_state():
    ns = lane_count("N1_J1") + lane_count("S1_J1")
    ew = lane_count("W_J1") + lane_count("J2_J1")

    return (bucket(ns), bucket(ew))


# ---------------- POLICY (GREEDY ONLY) ----------------
def choose_action(state):
    if state not in q_table:
        return 0
    return int(np.argmax(q_table[state]))


# ---------------- SAFE SIGNAL CONTROL ----------------
def run_yellow(phase):
    current = traci.trafficlight.getPhase(J1)

    if current != phase:
        traci.trafficlight.setPhase(J1, 1)  # yellow phase
        for _ in range(YELLOW_TIME):
            traci.simulationStep()

    traci.trafficlight.setPhase(J1, phase)


def apply_action(phase):
    run_yellow(phase)

    for _ in range(GREEN_TIME):
        traci.simulationStep()


# ---------------- TEST LOOP ----------------
def test():

    traci.start(["sumo", "-c", "simulation.sumocfg"])
    traci.simulationStep()

    state = get_state()

    total_wait = 0
    step_count = 0

    while traci.simulation.getMinExpectedNumber() > 0:

        action_idx = choose_action(state)
        phase = ACTION_SPACE[action_idx]

        apply_action(phase)

        state = get_state()

        # evaluation metric only (NO learning)
        for lane in traci.trafficlight.getControlledLanes(J1):
            total_wait += traci.lane.getWaitingTime(lane)

        step_count += 1

    traci.close()

    print("\n===== TEST RESULTS (J1 ONLY CONTROL) =====")
    print("Total waiting time:", total_wait)
    print("Average waiting per step:", total_wait / max(step_count, 1))


# ---------------- MAIN ----------------
if __name__ == "__main__":
    test()
    