import os
import sys
import random
import numpy as np
import pickle

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

import traci

# HYPERPARAMETERS 
ALPHA         = 0.1
GAMMA         = 0.95
EPSILON       = 1.0        # start fully random
EPSILON_DECAY = 0.98       # decay per episode (not per step)
MIN_EPSILON   = 0.05

EPISODES   = 150
GREEN_TIME = 10            # same value used in test.py
YELLOW_TIME = 3

J1 = "J1"
J2 = "J2"

# (j1_phase, j2_phase)  — only even phases = green phases
ACTION_SPACE = [
    (0, 0),
    (0, 2),
    (2, 0),
    (2, 2),
]

q_table = {}


# STATE

def bucket(x):
    """
    Finer bucketing — distinguishes low/mid/high queue better.
    5 -> 0 cars, 1 -> 1-2, 2 -> 3-6, 3 -> 7-12, 4 -> 13+
    """
    if x == 0:    return 0
    elif x <= 2:  return 1
    elif x <= 6:  return 2
    elif x <= 12: return 3
    else:         return 4


def get_halted(lane_id):
    """Halted (speed < 0.1 m/s) vehicles — better than presence count."""
    return traci.lane.getLastStepHaltingNumber(lane_id)


def get_state(j1_phase, j2_phase):
    """
    State = (j1_green_queue, j1_red_queue,
             j2_green_queue, j2_red_queue,
             j1_phase_bit,   j2_phase_bit)

    Keeping green vs red queues SEPARATE is the key improvement:
    the agent now knows WHICH direction has pressure, not just total load.
    phase_bit: 0 = NS green, 1 = EW green
    """

    # J1 — phase 0: NS green, phase 2: EW green
    if j1_phase == 0:
        j1_green = get_halted("N1_J1_0") + get_halted("S1_J1_0")
        j1_red   = get_halted("W_J1_0")  + get_halted("J2_J1_0")
    else:
        j1_green = get_halted("W_J1_0")  + get_halted("J2_J1_0")
        j1_red   = get_halted("N1_J1_0") + get_halted("S1_J1_0")

    # J2 — phase 0: NS green, phase 2: EW green
    if j2_phase == 0:
        j2_green = get_halted("N2_J2_0") + get_halted("S2_J2_0")
        j2_red   = get_halted("J1_J2_0") + get_halted("E_J2_0")
    else:
        j2_green = get_halted("J1_J2_0") + get_halted("E_J2_0")
        j2_red   = get_halted("N2_J2_0") + get_halted("S2_J2_0")

    return (
        bucket(j1_green), bucket(j1_red),
        bucket(j2_green), bucket(j2_red),
        j1_phase // 2,            # 0 or 1
        j2_phase // 2,            # 0 or 1
    )


#  REWARD

def get_reward():
    """
    Dense reward: penalize halted vehicles every step.
    Halted queue is sharper than waiting-time for traffic control.
    """
    total_halted = 0
    for lane in [
        "N1_J1_0", "S1_J1_0", "W_J1_0",  "J2_J1_0",
        "N2_J2_0", "S2_J2_0", "J1_J2_0", "E_J2_0",
    ]:
        total_halted += traci.lane.getLastStepHaltingNumber(lane)
    return -total_halted          # simple, unscaled, works well


# POLICY

def choose_action(state):
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTION_SPACE))
    if random.random() < EPSILON:
        return random.randint(0, len(ACTION_SPACE) - 1)
    return int(np.argmax(q_table[state]))


# ---------------- PHASE TRANSITION ----------------

# Yellow phase index for each green phase (from your tlLogic)
# phase 0 (NS green) -> yellow is phase 1
# phase 2 (EW green) -> yellow is phase 3
YELLOW_PHASE = {0: 1, 2: 3}
# phase 0 → NS green   (GGggrrrrGGggrrrr)
# phase 1 → NS yellow  (yyyyrrrryyyyrrrr)
# phase 2 → EW green   (rrrrGGggrrrrGGgg)
# phase 3 → EW yellow  (rrrryyyyrrrryyyy)

# if currently on phase 0 (NS green), yellow is phase 1
# if currently on phase 2 (EW green), yellow is phase 3

def apply_action(j1_new, j2_new, j1_cur, j2_cur):
    """
    Correct transition:
    1. Set yellow simultaneously for any junction that is switching.
    2. Advance simulation once for YELLOW_TIME steps.
    3. Set both junctions to their new green phase together.
    4. Run GREEN_TIME steps, collecting reward each step.
    """
    j1_switching = (j1_new != j1_cur)
    j2_switching = (j2_new != j2_cur)

    # Step 1 — yellow (simultaneous, NOT sequential)
    if j1_switching:
        traci.trafficlight.setPhase(J1, YELLOW_PHASE[j1_cur])
    if j2_switching:
        traci.trafficlight.setPhase(J2, YELLOW_PHASE[j2_cur])

    if j1_switching or j2_switching:
        for _ in range(YELLOW_TIME):
            traci.simulationStep()

    # Step 2 — green
    traci.trafficlight.setPhase(J1, j1_new)
    traci.trafficlight.setPhase(J2, j2_new)

    # Step 3 — run and collect reward each step
    cycle_reward = 0
    for _ in range(GREEN_TIME):
        traci.simulationStep()
        cycle_reward += get_reward()

    return cycle_reward


# Q UPDATE 

def update_q(state, action, reward, next_state):
    if next_state not in q_table:
        q_table[next_state] = np.zeros(len(ACTION_SPACE))

    best_next = np.max(q_table[next_state])
    td_error  = reward + GAMMA * best_next - q_table[state][action]
    q_table[state][action] += ALPHA * td_error


# TRAINING LOOP 

def train():
    global EPSILON

    for episode in range(EPISODES):

        traci.start(["sumo", "-c", "simulation.sumocfg", "--no-warnings"])
        traci.simulationStep()

        # both junctions start on NS green (phase 0)
        j1_phase = 0
        j2_phase = 0
        traci.trafficlight.setPhase(J1, j1_phase)
        traci.trafficlight.setPhase(J2, j2_phase)

        state        = get_state(j1_phase, j2_phase)
        total_reward = 0
        steps        = 0

        while traci.simulation.getMinExpectedNumber() > 0:

            action_idx      = choose_action(state)
            j1_new, j2_new  = ACTION_SPACE[action_idx]

            cycle_reward = apply_action(j1_new, j2_new, j1_phase, j2_phase)

            j1_phase   = j1_new
            j2_phase   = j2_new
            next_state = get_state(j1_phase, j2_phase)

            update_q(state, action_idx, cycle_reward, next_state)

            state        = next_state
            total_reward += cycle_reward
            steps        += 1

        traci.close()

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

        print(
            f"Ep {episode+1:3d} | "
            f"Steps: {steps:4d} | "
            f"Reward: {total_reward:8.1f} | "
            f"States: {len(q_table):4d} | "
            f"ε: {EPSILON:.3f}"
        )

    with open("qtable.pkl", "wb") as f:
        pickle.dump(q_table, f)

    print("\nTraining complete — qtable.pkl saved")


if __name__ == "__main__":
    train()