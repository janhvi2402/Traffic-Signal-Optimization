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
    (0, 0),
    (0, 2),
    (2, 0),
    (2, 2)
]

with open("qtable.pkl", "rb") as f:
    q_table = pickle.load(f)

print("States learned:", len(q_table))


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

traci.simulationStep()

cumulative_wait = 0
arrived = 0
step = 0

while traci.simulation.getMinExpectedNumber() > 0:

    step += 1

    for _ in range(GREEN_TIME):

        traci.simulationStep()

        arrived += len(
            traci.simulation.getArrivedIDList()
        )

        step_wait = 0

        for veh in traci.vehicle.getIDList():
            step_wait += traci.vehicle.getWaitingTime(veh)

        cumulative_wait += step_wait

    if step % 100 == 0:
        print(
            "Remaining:",
            traci.simulation.getMinExpectedNumber()
        )

traci.close()

print("\n===== RESULTS =====")
print("Vehicles Arrived :", arrived)
print("Cumulative Wait  :", cumulative_wait)