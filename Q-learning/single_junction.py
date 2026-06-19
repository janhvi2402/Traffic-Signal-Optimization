import os
import sys
import pickle
import numpy as np
import traci

# ---------------- SUMO SETUP ----------------
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")


# ---------------- CONFIG ----------------
J1 = "J1"
J2 = "J2"

GREEN_TIME = 6   # MUST match training
YELLOW_TIME = 3

ACTION_SPACE = [
    (0, 0),
    (0, 2),
    (2, 0),
    (2, 2)
]

# ---------------- LOAD Q TABLE ----------------
with open("qtable.pkl", "rb") as f:
    q_table = pickle.load(f)

print("Loaded states:", len(q_table))


# ---------------- STATE FUNCTION (MUST MATCH TRAINING) ----------------

def bucket(x):
    if x == 0:
        return 0
    elif x <= 3:
        return 1
    elif x <= 7:
        return 2
    elif x <= 12:
        return 3
    else:
        return 4


def lane_count(edge):
    return traci.lane.getLastStepVehicleNumber(edge + "_0")


def get_state():
    j1_ns = lane_count("N1_J1") + lane_count("S1_J1")
    j1_ew = lane_count("W_J1") + lane_count("J2_J1")

    j2_ns = lane_count("N2_J2") + lane_count("S2_J2")
    j2_ew = lane_count("J1_J2") + lane_count("E_J2")

    return (
        bucket(j1_ns),
        bucket(j1_ew),
        bucket(j2_ns),
        bucket(j2_ew)
    )


# ---------------- POLICY (GREEDY ONLY) ----------------

def choose_action(state):
    if state not in q_table:
        return 0
    return int(np.argmax(q_table[state]))


# ---------------- CONTROL ----------------

def run_cycle(j1_phase, j2_phase):

    traci.trafficlight.setPhase(J1, j1_phase)
    traci.trafficlight.setPhase(J2, j2_phase)

    for _ in range(GREEN_TIME):
        traci.simulationStep()


# ---------------- TEST ----------------

def test():

    traci.start(["sumo-gui", "-c", "simulation.sumocfg"])
    traci.simulationStep()

    state = get_state()

    total_wait = 0
    total_vehicles = 0
    steps = 0

    while traci.simulation.getMinExpectedNumber() > 0:

        action = choose_action(state)
        j1_phase, j2_phase = ACTION_SPACE[action]

        run_cycle(j1_phase, j2_phase)

        state = get_state()
        steps += 1

        # -------- METRICS --------
        total_vehicles += len(traci.simulation.getArrivedIDList())

        step_wait = 0
        for veh in traci.vehicle.getIDList():
            step_wait += traci.vehicle.getWaitingTime(veh)

        total_wait += step_wait

    traci.close()

    print("\n===== TEST RESULTS (BOTH JUNCTIONS CONTROLLED) =====")
    print("Steps:", steps)
    print("Vehicles Arrived:", total_vehicles)
    print("Cumulative Waiting Time:", total_wait)
    print("Avg Wait per Step:", total_wait / max(steps, 1))