import os
import sys
import csv
import random
import numpy as np
import pickle

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SUMOCFG_PATH = os.path.join(SCRIPT_DIR, "simulation.sumocfg")
QTABLE_PATH  = os.path.join(SCRIPT_DIR, "qtable.pkl")
TRAINLOG_PATH = os.path.join(SCRIPT_DIR, "training_log.csv")   # per-episode log for diagnostics

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

# NEW: minimum number of full green cycles a junction must hold before it's
# allowed to switch again. GREEN_TIME=10 steps already equals 10 seconds of
# sim time, so MIN_GREEN_CYCLES=1 enforces a 10-second minimum green — the
# junction can't switch again until at least one full cycle has elapsed.
# Without this, nothing stops the agent from switching back on the very
# next decision, so it never experiences (or gets reward feedback from)
# holding longer than one cycle even when the queue is still heavy.
MIN_GREEN_CYCLES = 1   # 1 cycle * GREEN_TIME(10) = 10s minimum green

# NEW: reward-shaping constants, kept small relative to the -total_halted
# term (which can easily be 10-30+ across 8 lanes) so they nudge behavior
# rather than dominate the queue signal. SWITCH_PENALTY discourages
# flip-flopping outright; WRONG_DIRECTION_PENALTY specifically discourages
# switching away from the arm that's still more congested than the arm
# gaining green. Tune these up if the agent still flip-flops too much, or
# down if it starts refusing to switch even when it clearly should.
SWITCH_PENALTY           = 0.5
WRONG_DIRECTION_PENALTY  = 1.0

J1 = "J1"
J2 = "J2"

# NEW: actions are now RELATIVE per junction (0 = stay, 1 = switch) instead
# of absolute target phases. This removes the old failure mode where a
# zero-initialized Q-table's argmax tie-break (always index 0) silently
# forced phase (0,0) regardless of current phase — with stay=0, a tie now
# defaults to holding the current phase instead of forcing a switch.
ACTION_SPACE = [(0, 0), (0, 1), (1, 0), (1, 1)]
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


def bucket_hold(cycles):
    """
    NEW: bucket how many green cycles the current phase has already been
    held for, so the state can distinguish "just switched" from "been
    holding a while" — must stay IDENTICAL to the copy in test.py.
    """
    if cycles == 0:   return 0
    elif cycles == 1: return 1
    elif cycles == 2: return 2
    else:             return 3


def get_halted(lane_id):
    return traci.lane.getLastStepHaltingNumber(lane_id)


def get_arm_queues(which, phase):
    """NEW: raw (unbucketed) halted counts for the currently-green / currently-red
    arm of a junction, given its phase. Factored out of get_state so reward
    shaping (get_reward) can reuse it for the wrong-direction check."""
    if which == "J1":
        if phase == 0:
            green = get_halted("N1_J1_0") + get_halted("S1_J1_0")
            red   = get_halted("W_J1_0")  + get_halted("J2_J1_0")
        else:
            green = get_halted("W_J1_0")  + get_halted("J2_J1_0")
            red   = get_halted("N1_J1_0") + get_halted("S1_J1_0")
    else:
        if phase == 0:
            green = get_halted("N2_J2_0") + get_halted("S2_J2_0")
            red   = get_halted("J1_J2_0") + get_halted("E_J2_0")
        else:
            green = get_halted("J1_J2_0") + get_halted("E_J2_0")
            red   = get_halted("N2_J2_0") + get_halted("S2_J2_0")
    return green, red


def get_state(j1_phase, j2_phase, j1_hold, j2_hold):
    j1_green, j1_red = get_arm_queues("J1", j1_phase)
    j2_green, j2_red = get_arm_queues("J2", j2_phase)

    return (
        bucket(j1_green), bucket(j1_red),
        bucket(j2_green), bucket(j2_red),
        j1_phase // 2,
        j2_phase // 2,
        bucket_hold(j1_hold),   # NEW
        bucket_hold(j2_hold),   # NEW
    )


def get_reward(j1_switched, j2_switched, j1_wrong_dir, j2_wrong_dir):
    """
    Normalized by GREEN_TIME so reward scale is comparable across
    different gt configs (raw cumulative halted count would otherwise
    scale with gt, making runs with different gt structurally
    incomparable).

    NEW: subtracts SWITCH_PENALTY per switch (discourages flip-flopping),
    plus WRONG_DIRECTION_PENALTY if the switch abandoned the arm that was
    still more congested than the arm gaining green. This directly targets
    the flat hold-duration-vs-backlog behaviour and ~72% flip-flop rate
    seen in diagnostics.
    """
    total_halted = 0
    for lane in [
        "N1_J1_0", "S1_J1_0", "W_J1_0",  "J2_J1_0",
        "N2_J2_0", "S2_J2_0", "J1_J2_0", "E_J2_0",
    ]:
        total_halted += traci.lane.getLastStepHaltingNumber(lane)

    reward = -total_halted
    if j1_switched:
        reward -= SWITCH_PENALTY
        if j1_wrong_dir:
            reward -= WRONG_DIRECTION_PENALTY
    if j2_switched:
        reward -= SWITCH_PENALTY
        if j2_wrong_dir:
            reward -= WRONG_DIRECTION_PENALTY
    return reward


def choose_action(state):
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTION_SPACE))
    if random.random() < EPSILON:
        return random.randint(0, len(ACTION_SPACE) - 1)
    return int(np.argmax(q_table[state]))


def apply_action(action_idx, j1_cur, j2_cur, j1_hold, j2_hold):
    """
    NEW: actions are relative (0=stay,1=switch) and masked by
    MIN_GREEN_CYCLES — a junction can only actually switch if the agent
    chose to AND it has held long enough. This forces the agent to sample
    longer holds during training instead of always taking the fastest
    possible switch, and gives get_reward the info it needs for the
    direction-correctness penalty.
    """
    j1_choice, j2_choice = ACTION_SPACE[action_idx]

    j1_switching = bool(j1_choice) and (j1_hold >= MIN_GREEN_CYCLES)
    j2_switching = bool(j2_choice) and (j2_hold >= MIN_GREEN_CYCLES)

    j1_new = (2 - j1_cur) if j1_switching else j1_cur   # 0<->2
    j2_new = (2 - j2_cur) if j2_switching else j2_cur

    # NEW: check, at the moment of switching, whether the arm losing green
    # was still MORE congested than the arm gaining it (raw counts, used
    # only for reward shaping — not part of the Q-table state).
    j1_wrong_dir = False
    j2_wrong_dir = False
    if j1_switching:
        g, r = get_arm_queues("J1", j1_cur)
        j1_wrong_dir = g > r
    if j2_switching:
        g, r = get_arm_queues("J2", j2_cur)
        j2_wrong_dir = g > r

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
        cycle_reward += get_reward(j1_switching, j2_switching, j1_wrong_dir, j2_wrong_dir)

    return cycle_reward / GREEN_TIME, j1_new, j2_new, j1_switching, j2_switching


def update_q(state, action, reward, next_state):
    global alpha
    if next_state not in q_table:
        q_table[next_state] = np.zeros(len(ACTION_SPACE))

    best_next = np.max(q_table[next_state])
    td_error  = reward + GAMMA * best_next - q_table[state][action]
    q_table[state][action] += alpha * td_error


def train():
    global EPSILON, alpha

    log_file = open(TRAINLOG_PATH, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["episode", "steps", "total_reward", "n_states", "epsilon", "alpha"])

    for episode in range(EPISODES):

        traci.start(["sumo", "-c", SUMOCFG_PATH, "--no-warnings"])
        traci.simulationStep()

        j1_phase = 0
        j2_phase = 0
        j1_hold  = 0   # NEW: cycles the current phase has been held
        j2_hold  = 0
        traci.trafficlight.setPhase(J1, j1_phase)
        traci.trafficlight.setPhaseDuration(J1, 9999)
        traci.trafficlight.setPhase(J2, j2_phase)
        traci.trafficlight.setPhaseDuration(J2, 9999)

        state        = get_state(j1_phase, j2_phase, j1_hold, j2_hold)
        total_reward = 0
        steps        = 0

        while traci.simulation.getMinExpectedNumber() > 0:
            action_idx = choose_action(state)

            cycle_reward, j1_new, j2_new, j1_switched, j2_switched = apply_action(
                action_idx, j1_phase, j2_phase, j1_hold, j2_hold
            )

            j1_phase = j1_new
            j2_phase = j2_new
            j1_hold  = 0 if j1_switched else j1_hold + 1   # NEW
            j2_hold  = 0 if j2_switched else j2_hold + 1   # NEW

            next_state = get_state(j1_phase, j2_phase, j1_hold, j2_hold)

            update_q(state, action_idx, cycle_reward, next_state)

            state         = next_state
            total_reward += cycle_reward
            steps        += 1

        traci.close()

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
        alpha   = max(ALPHA_MIN, alpha * ALPHA_DECAY)

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