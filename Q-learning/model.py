import os
import sys
import random
import numpy as np
import pickle

# SUMO setup
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

import traci

# Hyperparameters


ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2

EPISODES = 100

J1 = "J1"
J2 = "J2"

GREEN_TIME = 10

ACTION_SPACE = [
    (0, 0),
    (0, 2),
    (2, 0),
    (2, 2)
]

q_table = {}

# Helpers

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


def get_reward():

    total_wait = 0

    for tls in [J1, J2]:

        lanes = list(set(
            traci.trafficlight.getControlledLanes(tls)
        ))

        for lane in lanes:
            total_wait += traci.lane.getWaitingTime(lane)

    return -total_wait


def choose_action(state):

    if state not in q_table:
        q_table[state] = np.zeros(len(ACTION_SPACE))

    if random.random() < EPSILON:
        return random.randint(
            0,
            len(ACTION_SPACE)-1
        )

    return np.argmax(q_table[state])


def apply_action(action_idx):

    phase_j1, phase_j2 = ACTION_SPACE[action_idx]

    traci.trafficlight.setPhase(J1, phase_j1)
    traci.trafficlight.setPhase(J2, phase_j2)


# Training

def train():

    for episode in range(EPISODES):

        traci.start([
            "sumo",
            "-c",
            "simulation.sumocfg"
        ])
        traci.simulationStep()# after simulation step get the state

        state = get_state()

        total_reward = 0

        while traci.simulation.getMinExpectedNumber() > 0:

            action_idx = choose_action(state)

            apply_action(action_idx)

            for _ in range(GREEN_TIME):
                traci.simulationStep()

            reward = get_reward()

            next_state = get_state()

            if next_state not in q_table:
                q_table[next_state] = np.zeros(
                    len(ACTION_SPACE)
                )

            best_next = np.max(
                q_table[next_state]
            )

            q_table[state][action_idx] += (
                ALPHA *
                (
                    reward
                    + GAMMA * best_next
                    - q_table[state][action_idx]
                )
            )

            state = next_state

            total_reward += reward

        traci.close()

    print("\nTraining Complete")

    with open("qtable.pkl", "wb") as f:
        pickle.dump(q_table, f)

        print("Q-table saved")

        print(
            f"Episode {episode+1} "
            f"Reward = {total_reward:.2f}"
        )

    print("\nTraining Complete")

    print("\nLearned Q Table")

    for s, q in q_table.items():
        print(s, q)


if __name__ == "__main__":
    train()


