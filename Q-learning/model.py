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


# ---------------- HYPERPARAMETERS ----------------
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.3
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05

EPISODES = 100

J1 = "J1"
J2 = "J2"

GREEN_TIME = 6   # reduced (important for faster feedback)
YELLOW_TIME = 3

ACTION_SPACE = [
    (0, 0),
    (0, 2),
    (2, 0),
    (2, 2)
]

q_table = {}


# ---------------- STATE HELPERS ----------------

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


# ---------------- OPTIMIZED REWARD ----------------

def get_reward():
    total_wait = 0
    queue_pressure = 0

    for tls in [J1, J2]:
        lanes = set(traci.trafficlight.getControlledLanes(tls))

        for lane in lanes:
            wait = traci.lane.getWaitingTime(lane)
            veh = traci.lane.getLastStepVehicleNumber(lane)

            total_wait += wait
            queue_pressure += veh

    # normalized reward (IMPORTANT for stability)
    reward = -((total_wait * 0.7) + (queue_pressure * 0.3)) / 50.0

    return reward


# ---------------- POLICY ----------------

def choose_action(state):
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTION_SPACE))

    if random.random() < EPSILON:
        return random.randint(0, len(ACTION_SPACE) - 1)

    return int(np.argmax(q_table[state]))


# ---------------- SAFE TRANSITION ----------------

def run_yellow(tls_id, next_phase):
    current_phase = traci.trafficlight.getPhase(tls_id)

    if current_phase != next_phase:
        traci.trafficlight.setPhase(tls_id, 1)  # yellow
        for _ in range(YELLOW_TIME):
            traci.simulationStep()

    traci.trafficlight.setPhase(tls_id, next_phase)


def apply_action(j1_phase, j2_phase):
    run_yellow(J1, j1_phase)
    run_yellow(J2, j2_phase)

    for _ in range(GREEN_TIME):
        traci.simulationStep()


# ---------------- TRAINING LOOP ----------------

def train():

    global EPSILON

    for episode in range(EPISODES):

        traci.start(["sumo", "-c", "simulation.sumocfg"])
        traci.simulationStep()

        state = get_state()
        total_reward = 0

        while traci.simulation.getMinExpectedNumber() > 0:

            action_idx = choose_action(state)
            new_j1, new_j2 = ACTION_SPACE[action_idx]

            apply_action(new_j1, new_j2)

            reward = get_reward()
            next_state = get_state()

            if next_state not in q_table:
                q_table[next_state] = np.zeros(len(ACTION_SPACE))

            best_next = np.max(q_table[next_state])

            q_table[state][action_idx] += ALPHA * (
                reward + GAMMA * best_next - q_table[state][action_idx]
            )

            state = next_state
            total_reward += reward

        traci.close()

        # decay exploration
        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

        print(f"Episode {episode+1} | Reward: {total_reward:.2f} | EPS: {EPSILON:.3f}")

    # save Q-table
    with open("qtable.pkl", "wb") as f:
        pickle.dump(q_table, f)

    print("\nTraining Complete")
    print("Q-table saved")


# ---------------- MAIN ----------------

if __name__ == "__main__":
    train()