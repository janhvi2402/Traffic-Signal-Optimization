import os
import sys
import csv
import random
import numpy as np
import pickle

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SUMOCFG_PATH = os.path.join(SCRIPT_DIR, "simulation.sumocfg")
QTABLE_PATH  = os.path.join(SCRIPT_DIR, "qtable.pkl")
TRAINLOG_PATH = os.path.join(SCRIPT_DIR, "training_log.csv")   # NEW: per-episode log for diagnostics

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

import traci

# HYPERPARAMETERS
ALPHA_START   = 0.1
ALPHA_MIN     = 0.02
ALPHA_DECAY   = 0.995      # decay per episode — prevents late-training oscillation
GAMMA         = 0.95
EPSILON       = 1.0
EPSILON_DECAY = 0.98
MIN_EPSILON   = 0.05

EPISODES    = 300
GREEN_TIME  = 10
YELLOW_TIME = 3

J1 = "J1"
J2 = "J2"

ACTION_SPACE = [(0, 0), (0, 2), (2, 0), (2, 2)]
YELLOW_PHASE = {0: 1, 2: 3}


q_table = {}
alpha = ALPHA_START


def bucket(x):
    """
    NOTE: this function must stay IDENTICAL to the one in test.py.
    If they diverge, the Q-table's state keys won't match what test.py
    looks up, and every state will silently fall through as "unseen".

    Thresholds tuned to this project's route file (~0.6-0.7 veh/sec
    per entry arm, ~3.8 veh/sec network-wide) — queues regularly exceed
    12 vehicles under this demand, so the old 5-bucket version collapsed
    most of the congested / interesting range into a single bucket.
    """
    if x == 0:     return 0
    elif x <= 2:   return 1
    elif x <= 5:   return 2
    elif x <= 9:   return 3
    elif x <= 15:  return 4
    elif x <= 25:  return 5
    else:          return 6


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


def get_reward():
    """
    Normalized by GREEN_TIME so reward scale is comparable across
    different gt configs (raw cumulative halted count would otherwise
    scale with gt, making runs with different gt structurally
    incomparable).
    """
    total_halted = 0
    for lane in [
        "N1_J1_0", "S1_J1_0", "W_J1_0",  "J2_J1_0",
        "N2_J2_0", "S2_J2_0", "J1_J2_0", "E_J2_0",
    ]:
        total_halted += traci.lane.getLastStepHaltingNumber(lane)
    return -total_halted


def choose_action(state):
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTION_SPACE))
    if random.random() < EPSILON:
        return random.randint(0, len(ACTION_SPACE) - 1)
    return int(np.argmax(q_table[state]))


def apply_action(j1_new, j2_new, j1_cur, j2_cur):
    j1_switching = (j1_new != j1_cur)
    j2_switching = (j2_new != j2_cur)

    if j1_switching:
        traci.trafficlight.setPhase(J1, YELLOW_PHASE[j1_cur])
        traci.trafficlight.setPhaseDuration(J1, 9999)
    if j2_switching:
        traci.trafficlight.setPhase(J2, YELLOW_PHASE[j2_cur])
        traci.trafficlight.setPhaseDuration(J2, 9999)

    if j1_switching or j2_switching:
        for _ in range(YELLOW_TIME):
            traci.simulationStep()

    traci.trafficlight.setPhase(J1, j1_new)
    traci.trafficlight.setPhaseDuration(J1, 9999)
    traci.trafficlight.setPhase(J2, j2_new)
    traci.trafficlight.setPhaseDuration(J2, 9999)

    cycle_reward = 0
    for _ in range(GREEN_TIME):
        traci.simulationStep()
        cycle_reward += get_reward()

    return cycle_reward / GREEN_TIME   # normalized per-step reward


def update_q(state, action, reward, next_state):
    global alpha
    if next_state not in q_table:
        q_table[next_state] = np.zeros(len(ACTION_SPACE))

    best_next = np.max(q_table[next_state])
    td_error  = reward + GAMMA * best_next - q_table[state][action]
    q_table[state][action] += alpha * td_error


def train():
    global EPSILON, alpha

    # NEW: fresh training log for this run, used by diagnostic.py
    log_file = open(TRAINLOG_PATH, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["episode", "steps", "total_reward", "n_states", "epsilon", "alpha"])

    for episode in range(EPISODES):

        traci.start(["sumo", "-c", SUMOCFG_PATH, "--no-warnings"])
        traci.simulationStep()

        j1_phase = 0
        j2_phase = 0
        traci.trafficlight.setPhase(J1, j1_phase)
        traci.trafficlight.setPhaseDuration(J1, 9999)
        traci.trafficlight.setPhase(J2, j2_phase)
        traci.trafficlight.setPhaseDuration(J2, 9999)

        state        = get_state(j1_phase, j2_phase)
        total_reward = 0
        steps        = 0

        while traci.simulation.getMinExpectedNumber() > 0:
            action_idx     = choose_action(state)
            j1_new, j2_new = ACTION_SPACE[action_idx]

            cycle_reward = apply_action(j1_new, j2_new, j1_phase, j2_phase)

            j1_phase   = j1_new
            j2_phase   = j2_new
            next_state = get_state(j1_phase, j2_phase)

            update_q(state, action_idx, cycle_reward, next_state)

            state         = next_state
            total_reward += cycle_reward
            steps        += 1

        traci.close()

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
        alpha   = max(ALPHA_MIN, alpha * ALPHA_DECAY)

        # NEW: log this episode's stats for later convergence plots
        log_writer.writerow([episode + 1, steps, total_reward, len(q_table), EPSILON, alpha])
        log_file.flush()

        print(
            f"Ep {episode+1:3d} | Steps: {steps:4d} | Reward: {total_reward:8.2f} | "
            f"States: {len(q_table):4d} | eps: {EPSILON:.3f} | alpha: {alpha:.4f}"
        )

    log_file.close()

    with open(QTABLE_PATH, "wb") as f:
        pickle.dump(q_table, f)

    print(f"\nTraining complete — qtable.pkl saved. Final state coverage: {len(q_table)} states")
    print(f"Per-episode log saved to {TRAINLOG_PATH}")


if __name__ == "__main__":
    train()