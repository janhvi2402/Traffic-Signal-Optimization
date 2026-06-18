import os
import sys
import pickle
import numpy as np
import traci

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

J1 = "J1"
J2 = "J2"

GREEN_TIME = 10

ACTION_SPACE = [
    (0,0),
    (0,2),
    (2,0),
    (2,2)
]

with open("qtable.pkl", "rb") as f:
    q_table = pickle.load(f)


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
    lane_id = edge + "_0"
    return traci.lane.getLastStepVehicleNumber(lane_id)

# (
#  J1_NS,
#  J1_EW,
#  J2_NS,
#  J2_EW
# )
def get_state():

    j1_ns = (
        lane_count("N1_J1")
        + lane_count("S1_J1")
    )

    j1_ew = (
        lane_count("W_J1")
        + lane_count("J2_J1")
    )

    j2_ns = (
        lane_count("N2_J2")
        + lane_count("S2_J2")
    )

    j2_ew = (
        lane_count("J1_J2")
        + lane_count("E_J2")
    )

    return (
        bucket(j1_ns),
        bucket(j1_ew),
        bucket(j2_ns),
        bucket(j2_ew)
    )

traci.start([
    "sumo-gui",
    "-c",
    "simulation.sumocfg"
])

total_wait = 0

while traci.simulation.getMinExpectedNumber() > 0:

    state = get_state()

    if state in q_table:
        action_idx = np.argmax(q_table[state])
    else:
        action_idx = 0

    phase_j1, phase_j2 = ACTION_SPACE[action_idx]

    traci.trafficlight.setPhase(J1, phase_j1)
    traci.trafficlight.setPhase(J2, phase_j2)

    for _ in range(GREEN_TIME):

        traci.simulationStep()

        for tls in [J1, J2]:

            lanes = list(set(
                traci.trafficlight.getControlledLanes(tls)
            ))

            for lane in lanes:
                total_wait += traci.lane.getWaitingTime(lane)

traci.close()

print("Total Waiting Time:", total_wait)