"""
plot_switch_timing_qlearning.py
==============================================================
Q-learning counterpart to plot_switch_timing.py (the PPO version).

Runs several evaluation episodes using the trained qtable.pkl
(same greedy-action logic as test.py), and instead of just plotting
*when* switches happen, digs into *why*:

  1. Real hold duration per switch (steps between consecutive
     switches of the same junction, computed WITHIN each episode --
     pooling across episodes without resetting at episode boundaries
     silently corrupts this number, since step counters restart at 0
     each episode).
  2. Flip-flop rate: % of switches that happen at the theoretical
     minimum interval (YELLOW_TIME + GREEN_TIME) -- i.e. the agent
     switches back again on the very next decision. A high rate here
     means the policy is toggling on a fixed rhythm rather than
     holding green while a queue clears.
  3. Premature-switch rate: at the moment of each switch, was the arm
     LOSING green still more congested (higher halted count) than the
     arm GAINING green? If this happens often, the agent is not
     conditioning on queue imbalance -- it's switching on a
     timer-like/shortcut pattern that happened to correlate with
     reward during training.

Run from the Q-learning project root:
    python plot_switch_timing_qlearning.py
==============================================================
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not set")

import traci

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR   = os.path.join(SCRIPT_DIR, "..", "common")
sys.path.append(COMMON_DIR)

SUMOCFG_PATH = os.path.join(SCRIPT_DIR, "simulation.sumocfg")
QTABLE_PATH  = os.path.join(SCRIPT_DIR, "qtable.pkl")

# Pull the ACTUAL training hyperparameters off train.py rather than
# retyping them -- keeps the label from drifting out of sync with
# whatever qtable.pkl this run is actually analyzing.
import model as trainmod

N_EPISODES = 5
BIN_SIZE   = 100
TL_IDS     = ["J1", "J2"]
MIN_INTERVAL = trainmod.YELLOW_TIME + trainmod.GREEN_TIME   # fastest possible switch-back

RUN_LABEL = "qlearning_v1"   # <-- update this each time you retrain


def bucket(x):
    """Must stay identical to bucket() in train.py / test.py."""
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


def get_arm_queues(tl_phase, which):
    """Raw (unbucketed) halted counts for the currently-green / currently-red
    arm of a junction, given its phase. Used to check queue-justification
    of switches, independent of the coarse bucket() the policy trains on."""
    if which == "J1":
        if tl_phase == 0:
            green = get_halted("N1_J1_0") + get_halted("S1_J1_0")
            red   = get_halted("W_J1_0")  + get_halted("J2_J1_0")
        else:
            green = get_halted("W_J1_0")  + get_halted("J2_J1_0")
            red   = get_halted("N1_J1_0") + get_halted("S1_J1_0")
    else:
        if tl_phase == 0:
            green = get_halted("N2_J2_0") + get_halted("S2_J2_0")
            red   = get_halted("J1_J2_0") + get_halted("E_J2_0")
        else:
            green = get_halted("J1_J2_0") + get_halted("E_J2_0")
            red   = get_halted("N2_J2_0") + get_halted("S2_J2_0")
    return green, red


def run_episode(q_table, seed):
    """Same control loop as test.py's run_qlearning_episode, but records,
    for every switch decision: the sim step, and the green/red arm queue
    counts of that junction at the moment the switch was decided (i.e.
    right before the yellow phase begins)."""
    traci.start(["sumo", "-c", SUMOCFG_PATH, "--no-warnings", "--seed", str(seed)])
    traci.simulationStep()

    j1_phase = 0
    j2_phase = 0
    traci.trafficlight.setPhase(trainmod.J1, j1_phase)
    traci.trafficlight.setPhaseDuration(trainmod.J1, 9999)
    traci.trafficlight.setPhase(trainmod.J2, j2_phase)
    traci.trafficlight.setPhaseDuration(trainmod.J2, 9999)

    sim_steps = 0
    switch_events = {"J1": [], "J2": []}   # list of dicts: step, green_q, red_q

    while traci.simulation.getMinExpectedNumber() > 0:
        state = get_state(j1_phase, j2_phase)
        if state in q_table:
            action_idx = int(np.argmax(q_table[state]))
        else:
            action_idx = 0

        j1_new, j2_new = trainmod.ACTION_SPACE[action_idx]
        j1_switching = (j1_new != j1_phase)
        j2_switching = (j2_new != j2_phase)

        if j1_switching:
            g, r = get_arm_queues(j1_phase, "J1")
            switch_events["J1"].append({"step": sim_steps, "green_q": g, "red_q": r})
            traci.trafficlight.setPhase(trainmod.J1, trainmod.YELLOW_PHASE[j1_phase])
            traci.trafficlight.setPhaseDuration(trainmod.J1, 9999)
        if j2_switching:
            g, r = get_arm_queues(j2_phase, "J2")
            switch_events["J2"].append({"step": sim_steps, "green_q": g, "red_q": r})
            traci.trafficlight.setPhase(trainmod.J2, trainmod.YELLOW_PHASE[j2_phase])
            traci.trafficlight.setPhaseDuration(trainmod.J2, 9999)

        if j1_switching or j2_switching:
            for _ in range(trainmod.YELLOW_TIME):
                traci.simulationStep()
                sim_steps += 1

        traci.trafficlight.setPhase(trainmod.J1, j1_new)
        traci.trafficlight.setPhaseDuration(trainmod.J1, 9999)
        traci.trafficlight.setPhase(trainmod.J2, j2_new)
        traci.trafficlight.setPhaseDuration(trainmod.J2, 9999)
        j1_phase, j2_phase = j1_new, j2_new

        for _ in range(trainmod.GREEN_TIME):
            traci.simulationStep()
            sim_steps += 1

    traci.close()
    return switch_events, sim_steps


def main():
    if not os.path.exists(QTABLE_PATH):
        sys.exit(f"FATAL: qtable.pkl not found at {QTABLE_PATH}")
    with open(QTABLE_PATH, "rb") as f:
        q_table = pickle.load(f)

    config_str = (
        f"alpha_start={trainmod.ALPHA_START}, alpha_min={trainmod.ALPHA_MIN}, "
        f"alpha_decay={trainmod.ALPHA_DECAY}, gamma={trainmod.GAMMA}, "
        f"epsilon_start={trainmod.EPSILON}, epsilon_decay={trainmod.EPSILON_DECAY}, "
        f"min_epsilon={trainmod.MIN_EPSILON}\n"
        f"episodes={trainmod.EPISODES}, green_time={trainmod.GREEN_TIME}, "
        f"yellow_time={trainmod.YELLOW_TIME}, states_learned={len(q_table)}, "
        f"min_possible_interval={MIN_INTERVAL} steps"
    )

    all_switch_steps  = {tl: [] for tl in TL_IDS}   # for the timing histogram (pooled ok, event *counts* aren't affected by episode resets)
    all_hold_durations = {tl: [] for tl in TL_IDS}  # correctly computed WITHIN-episode diffs only
    all_events         = {tl: [] for tl in TL_IDS}  # full event dicts, for premature-switch check
    max_sim_steps = 0

    for ep in range(N_EPISODES):
        switch_events, sim_steps = run_episode(q_table, seed=ep)
        max_sim_steps = max(max_sim_steps, sim_steps)
        for tl in TL_IDS:
            events = switch_events[tl]
            all_switch_steps[tl].extend(e["step"] for e in events)
            all_events[tl].extend(events)
            # hold durations: diffs WITHIN this episode only -- never across
            # the episode boundary, since step counters reset to 0 each episode
            steps_this_ep = [e["step"] for e in events]
            if len(steps_this_ep) >= 2:
                all_hold_durations[tl].extend(np.diff(steps_this_ep).tolist())
        print(f"episode {ep} done — J1: {len(switch_events['J1'])} switches, "
              f"J2: {len(switch_events['J2'])} switches, sim_steps={sim_steps}")

    bins = np.arange(0, max_sim_steps + BIN_SIZE, BIN_SIZE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    fig.suptitle(
        f"Switch timing & policy sanity (Q-learning) — {RUN_LABEL}\n{config_str}\ngenerated {timestamp}",
        fontsize=10,
    )

    insights = []

    for col, tl in enumerate(TL_IDS):
        steps = all_switch_steps[tl]
        holds = np.array(all_hold_durations[tl])
        events = all_events[tl]

        # (row 0) timing histogram, as before
        ax = axes[0, col]
        ax.hist(steps, bins=bins, color="steelblue", edgecolor="white")
        ax.set_title(f"{tl}: switch timing (n={len(steps)} switches / {N_EPISODES} eps)")
        ax.set_xlabel("simulation step within episode")
        ax.set_ylabel("switch count per 100-step bin")

        # (row 1) hold-duration histogram -- the actually meaningful one
        ax = axes[1, col]
        if len(holds) > 0:
            max_hold = int(holds.max())
            hbins = np.arange(0, max_hold + trainmod.GREEN_TIME, trainmod.GREEN_TIME)
            ax.hist(holds, bins=hbins, color="darkorange", edgecolor="white")
            ax.axvline(MIN_INTERVAL, color="crimson", linestyle="--",
                       label=f"min possible = {MIN_INTERVAL}")
            ax.axvline(np.median(holds), color="navy", linestyle="-",
                       label=f"median = {np.median(holds):.0f}")
            ax.legend(fontsize=8)
        ax.set_title(f"{tl}: hold duration between switches (real, per-episode)")
        ax.set_xlabel("steps held before next switch")
        ax.set_ylabel("count")

        # (row 2) summary text panel
        ax = axes[2, col]
        ax.axis("off")
        if len(holds) > 0 and len(events) > 0:
            flip_flop_rate = 100 * np.mean(holds == MIN_INTERVAL)
            near_min_rate  = 100 * np.mean(holds <= MIN_INTERVAL + trainmod.GREEN_TIME)  # min or one cycle over
            premature = np.array([e["green_q"] > e["red_q"] for e in events])
            premature_rate = 100 * premature.mean()
            mean_green_q = np.mean([e["green_q"] for e in events])
            mean_red_q   = np.mean([e["red_q"] for e in events])

            summary = (
                f"Mean hold duration : {holds.mean():.1f} steps  (median {np.median(holds):.0f})\n"
                f"Min possible hold  : {MIN_INTERVAL} steps\n"
                f"Flip-flop rate     : {flip_flop_rate:.1f}%  (switches back at min interval)\n"
                f"Near-min hold rate : {near_min_rate:.1f}%  (<= min + 1 cycle)\n\n"
                f"Mean queue on arm LOSING green at switch : {mean_green_q:.1f}\n"
                f"Mean queue on arm GAINING green at switch: {mean_red_q:.1f}\n"
                f"Premature-switch rate : {premature_rate:.1f}%\n"
                f"  (switched away while losing-green arm\n"
                f"   still MORE congested than gaining arm)"
            )
            ax.text(0.02, 0.95, summary, va="top", ha="left", fontsize=9.5, family="monospace")

            insights.append(
                f"{tl}: mean hold {holds.mean():.1f} steps, flip-flop rate {flip_flop_rate:.1f}%, "
                f"premature-switch rate {premature_rate:.1f}%."
            )
            if flip_flop_rate > 30:
                insights.append(
                    f"  WARNING ({tl}): >{flip_flop_rate:.0f}% of switches happen at the minimum "
                    f"possible interval -- policy looks like it's toggling on a fixed rhythm "
                    f"rather than holding green while a queue clears."
                )
            if premature_rate > 40:
                insights.append(
                    f"  WARNING ({tl}): {premature_rate:.0f}% of switches abandon the more-congested "
                    f"arm -- the policy does not appear to be conditioning on queue imbalance."
                )
        else:
            ax.text(0.5, 0.5, "not enough switch events", ha="center", va="center")

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_filename = f"switch_timing_plot_qlearning_{RUN_LABEL}_{timestamp}.png"
    out_path = os.path.join(SCRIPT_DIR, out_filename)
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot -> {out_path}")

    print(f"\nConfig: {config_str}\n")
    print("\n".join(insights))


if __name__ == "__main__":
    main()