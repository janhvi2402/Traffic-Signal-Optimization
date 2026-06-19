import os
import sys
import numpy as np
import pickle

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

import traci

GREEN_TIME  = 10     # MUST match train.py
YELLOW_TIME = 3

J1 = "J1"
J2 = "J2"

ACTION_SPACE = [(0,0),(0,2),(2,0),(2,2)]
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


def apply_action(j1_new, j2_new, j1_cur, j2_cur):
    if j1_new != j1_cur:
        traci.trafficlight.setPhase(J1, YELLOW_PHASE[j1_cur])
    if j2_new != j2_cur:
        traci.trafficlight.setPhase(J2, YELLOW_PHASE[j2_cur])

    if j1_new != j1_cur or j2_new != j2_cur:
        for _ in range(YELLOW_TIME):
            traci.simulationStep()

    traci.trafficlight.setPhase(J1, j1_new)
    traci.trafficlight.setPhase(J2, j2_new)

    for _ in range(GREEN_TIME):
        traci.simulationStep()


traci.start(["sumo-gui", "-c", "simulation.sumocfg"])
traci.simulationStep()

j1_phase = 0
j2_phase = 0
traci.trafficlight.setPhase(J1, j1_phase)
traci.trafficlight.setPhase(J2, j2_phase)

cumulative_wait = 0
arrived         = 0
sim_steps          = 0

# inside your test while loop, replace the apply_action call with this pattern:

while traci.simulation.getMinExpectedNumber() > 0:

    # --- decide action (every GREEN_TIME steps) ---
    state = get_state(j1_phase, j2_phase)
    action_idx = int(np.argmax(q_table[state])) if state in q_table else 0
    j1_new, j2_new = ACTION_SPACE[action_idx]

    # --- yellow transition ---
    if j1_new != j1_phase:
        traci.trafficlight.setPhase(J1, YELLOW_PHASE[j1_phase])
    if j2_new != j2_phase:
        traci.trafficlight.setPhase(J2, YELLOW_PHASE[j2_phase])

    if j1_new != j1_phase or j2_new != j2_phase:
        for _ in range(YELLOW_TIME):
            traci.simulationStep()
            sim_steps+= 1
            arrived += len(traci.simulation.getArrivedIDList())
            for veh in traci.vehicle.getIDList():
                cumulative_wait += traci.vehicle.getWaitingTime(veh)

    # --- green phase ---
    traci.trafficlight.setPhase(J1, j1_new)
    traci.trafficlight.setPhase(J2, j2_new)
    j1_phase, j2_phase = j1_new, j2_new

    for _ in range(GREEN_TIME):
        traci.simulationStep()
        sim_steps+= 1
        arrived += len(traci.simulation.getArrivedIDList())
        for veh in traci.vehicle.getIDList():
            cumulative_wait += traci.vehicle.getWaitingTime(veh)

traci.close()

print("\n===== TEST RESULTS =====")
print(f"sim_sim_steps            : {sim_steps}")
print(f"Vehicles Arrived  : {arrived}")
print(f"Cumulative Wait   : {cumulative_wait:.0f}s")
print(f"Avg Wait/Step     : {cumulative_wait/max(sim_steps,1):.2f}s")