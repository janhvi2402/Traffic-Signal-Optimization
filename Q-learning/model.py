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

# State space is 8^4 (queue buckets) * 2*2 (phase feats) * 4*4 (hold
# buckets, see bucket_hold below) = 262,144 possible keys -- far fewer
# will actually be visited, but this is bigger than a 300-episode /
# 0.98-decay schedule was tuned for, so decay is slowed and episode count
# raised moderately (not as aggressively as the earlier 600-episode
# attempt, since GREEN_TIME=5 already doubles the number of decision
# points sampled per episode compared to GREEN_TIME=10).
EPSILON_DECAY = 0.985
MIN_EPSILON   = 0.05
# CHANGED: 450 -> 500. bucket_hold went from 3 values to 4 (see below, fixes
# a state-aliasing bug), which grew the nominal state space ~1.8x -- small
# bump to give the table a bit more room to converge.
EPISODES    = 500

# GREEN_TIME halved from the original 10s -> 5s. The win is granularity:
# hold durations beyond the min-green floor now extend in 5s steps
# (10, 15, 20, 25...) instead of 10s jumps, giving the agent finer control
# for matching hold length to queue size. Reward is normalized by
# GREEN_TIME (see get_reward/apply_action), so switch/direction penalties
# end up the same effective magnitude per switch regardless of this value
# -- verified below, not a source of drift.
GREEN_TIME  = 5
YELLOW_TIME = 3
# MIN_GREEN_CYCLES=2 * GREEN_TIME(5) = 10s minimum green -- same floor as
# the very first version of this project. A junction with an empty arm can
# still switch away at 10s; this does NOT force a longer wait just because
# the state representation elsewhere got richer.
MIN_GREEN_CYCLES = 2

# Reward-shaping constants, kept small relative to the -total_halted term.
# SWITCH_PENALTY is a flat tax on churn; WRONG_DIRECTION_PENALTY is
# multiplied by the bucketed size of the abandoned queue (see get_reward)
# so abandoning a big backlog costs more than abandoning a small one --
# this is the main lever that should make hold duration track queue size.
SWITCH_PENALTY           = 0.5
WRONG_DIRECTION_PENALTY  = 1.0   # multiplied by bucket(abandoned queue), see get_reward

J1 = "J1"
J2 = "J2"

# Actions are RELATIVE per junction (0 = stay, 1 = switch) instead of
# absolute target phases -- with stay=0, a zero-initialized Q-table's
# argmax tie-break defaults to holding the current phase, not forcing a
# switch.
ACTION_SPACE = [(0, 0), (0, 1), (1, 0), (1, 1)]
YELLOW_PHASE = {0: 1, 2: 3}


q_table = {}
alpha = ALPHA_START


def bucket(x):
    """
    NOTE: this function must stay IDENTICAL to the one in test.py.
    If they diverge, the Q-table's state keys won't match what test.py
    looks up, and every state will silently fall through as "unseen".

    Sanity-checked against network.net.xml: every arm this sums is a
    single-lane edge of length ~92.80m (or 85.60m for the two
    inter-junction links), so at SUMO's default ~7.5m/vehicle spacing a
    green/red arm-pair sum tops out around 23-24 vehicles. These 8
    buckets (0, 1-2, 3-5, 6-8, 9-11, 12-15, 16-20, 21+) cover that full
    range with the finest resolution in the 3-15 zone where most
    switch/hold decisions actually happen.
    """
    if x == 0:     return 0
    elif x <= 2:   return 1
    elif x <= 5:   return 2
    elif x <= 8:   return 3
    elif x <= 11:  return 4
    elif x <= 15:  return 5
    elif x <= 20:  return 6
    else:          return 7


def bucket_hold(cycles):
    """
    CHANGED: was 3 values (<=floor, <=floor+2, >floor+2). That collapsed
    hold=0, hold=1, AND hold=MIN_GREEN_CYCLES into a single bucket 0 --
    but hold < MIN_GREEN_CYCLES is a state where "switch" is masked into a
    no-op regardless of what the agent picks, while hold == MIN_GREEN_CYCLES
    is the first cycle where switching actually does something. Merging
    those muddies the Q-values right at the moment the decision starts to
    matter. Now split into 4: forced-stay / floor-just-reached / extended
    a bit / extended a lot. MUST stay IDENTICAL to the copy in test.py.
    """
    if cycles < MIN_GREEN_CYCLES:         return 0   # forced to stay, floor not reached
    elif cycles == MIN_GREEN_CYCLES:      return 1   # floor just reached, free to act
    elif cycles <= MIN_GREEN_CYCLES + 2:  return 2   # extended a bit past floor
    else:                                  return 3   # extended a lot


def get_halted(lane_id):
    return traci.lane.getLastStepHaltingNumber(lane_id)


def get_arm_queues(which, phase):
    """Raw (unbucketed) halted counts for the currently-green / currently-red
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
        bucket_hold(j1_hold),
        bucket_hold(j2_hold),
    )


def get_reward(j1_switched, j2_switched, j1_wrong_dir, j2_wrong_dir,
                j1_abandoned_bucket, j2_abandoned_bucket):
    """
    Normalized by GREEN_TIME so reward scale is comparable across
    different gt configs.

    WRONG_DIRECTION_PENALTY is multiplied by the bucketed size of the
    queue that was abandoned (0-7), instead of applied as a flat constant:
    abandoning a small residual queue is cheap, abandoning a heavy backlog
    is expensive. This is the mechanism meant to teach "hold longer when
    the queue you're serving is still big". Because this constant is added
    every step of the cycle loop and the total is divided by GREEN_TIME at
    the end, its effective per-switch magnitude is invariant to GREEN_TIME.
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
            reward -= WRONG_DIRECTION_PENALTY * j1_abandoned_bucket
    if j2_switched:
        reward -= SWITCH_PENALTY
        if j2_wrong_dir:
            reward -= WRONG_DIRECTION_PENALTY * j2_abandoned_bucket
    return reward


def choose_action(state):
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTION_SPACE))
    if random.random() < EPSILON:
        return random.randint(0, len(ACTION_SPACE) - 1)
    return int(np.argmax(q_table[state]))


def apply_action(action_idx, j1_cur, j2_cur, j1_hold, j2_hold):
    """
    Actions are relative (0=stay,1=switch) and masked by MIN_GREEN_CYCLES —
    a junction can only actually switch if the agent chose to AND it has
    held long enough.
    """
    j1_choice, j2_choice = ACTION_SPACE[action_idx]

    j1_switching = bool(j1_choice) and (j1_hold >= MIN_GREEN_CYCLES)
    j2_switching = bool(j2_choice) and (j2_hold >= MIN_GREEN_CYCLES)

    j1_new = (2 - j1_cur) if j1_switching else j1_cur   # 0<->2
    j2_new = (2 - j2_cur) if j2_switching else j2_cur

    # Check, at the moment of switching, whether the arm losing green was
    # still MORE congested than the arm gaining it, and how big that
    # abandoned queue was (bucketed). Used only for reward shaping — not
    # part of the Q-table state.
    j1_wrong_dir = False
    j2_wrong_dir = False
    j1_abandoned_bucket = 0
    j2_abandoned_bucket = 0
    if j1_switching:
        g, r = get_arm_queues("J1", j1_cur)
        j1_wrong_dir = g > r
        j1_abandoned_bucket = bucket(g)
    if j2_switching:
        g, r = get_arm_queues("J2", j2_cur)
        j2_wrong_dir = g > r
        j2_abandoned_bucket = bucket(g)

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
        cycle_reward += get_reward(
            j1_switching, j2_switching, j1_wrong_dir, j2_wrong_dir,
            j1_abandoned_bucket, j2_abandoned_bucket
        )

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
        j1_hold  = 0   # cycles the current phase has been held
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
            j1_hold  = 0 if j1_switched else j1_hold + 1
            j2_hold  = 0 if j2_switched else j2_hold + 1

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