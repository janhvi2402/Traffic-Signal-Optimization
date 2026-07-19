"""
plot_switch_timing_qlearning.py
==============================================================
Q-learning counterpart to plot_switch_timing.py (the PPO version).

Runs several evaluation episodes using the trained qtable.pkl
(same policy/greedy-action logic as test.py), logs the simulation
step at which each junction switches phase, and plots a histogram
of switch timing over the episode for J1 and J2.

The plot title is labeled with the actual hyperparameters pulled
live from train.py (ALPHA/GAMMA/EPSILON schedule, EPISODES, GREEN_TIME,
YELLOW_TIME) so you never lose track of which trained policy a given
plot belongs to -- same idea as the PPO script pulling switch_penalty /
wrong_direction_penalty / MIN_GREEN off the loaded env instance.

Also reports, per junction, the average number of simulation steps
between consecutive switches (i.e. how often the light actually
changes phase under the learned policy).

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
# retyping them here -- avoids the label silently drifting out of sync
# with what the loaded qtable.pkl was actually trained with.
import train as trainmod

N_EPISODES = 5
BIN_SIZE   = 100
TL_IDS     = ["J1", "J2"]

# NEW: label this run manually if you want a custom note (e.g. which
# qtable checkpoint or bucket scheme) -- combined automatically with
# the live hyperparameters read from train.py below.
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


def run_episode(q_table, seed):
    """
    Same control loop as test.py's run_qlearning_episode, but instead
    of accumulating wait-time stats it records the sim_steps index at
    the moment each junction is *decided* to switch (i.e. right before
    the yellow phase begins), so switch timing lines up with real
    simulation time rather than decision-cycle count.
    """
    traci.start(["sumo", "-c", SUMOCFG_PATH, "--no-warnings", "--seed", str(seed)])
    traci.simulationStep()

    j1_phase = 0
    j2_phase = 0
    traci.trafficlight.setPhase(trainmod.J1, j1_phase)
    traci.trafficlight.setPhaseDuration(trainmod.J1, 9999)
    traci.trafficlight.setPhase(trainmod.J2, j2_phase)
    traci.trafficlight.setPhaseDuration(trainmod.J2, 9999)

    sim_steps = 0
    switch_steps = {"J1": [], "J2": []}

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
            switch_steps["J1"].append(sim_steps)
            traci.trafficlight.setPhase(trainmod.J1, trainmod.YELLOW_PHASE[j1_phase])
            traci.trafficlight.setPhaseDuration(trainmod.J1, 9999)
        if j2_switching:
            switch_steps["J2"].append(sim_steps)
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
    return switch_steps, sim_steps


def mean_switch_interval(steps):
    """Average number of sim steps between consecutive switches."""
    if len(steps) < 2:
        return float("nan")
    diffs = np.diff(sorted(steps))
    return float(np.mean(diffs))


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
        f"yellow_time={trainmod.YELLOW_TIME}, states_learned={len(q_table)}"
    )

    all_switch_steps = {tl: [] for tl in TL_IDS}
    per_episode_intervals = {tl: [] for tl in TL_IDS}
    max_sim_steps = 0

    for ep in range(N_EPISODES):
        switch_steps, sim_steps = run_episode(q_table, seed=ep)
        max_sim_steps = max(max_sim_steps, sim_steps)
        for tl in TL_IDS:
            all_switch_steps[tl].extend(switch_steps[tl])
            per_episode_intervals[tl].append(mean_switch_interval(switch_steps[tl]))
        print(f"episode {ep} done — J1: {len(switch_steps['J1'])} switches, "
              f"J2: {len(switch_steps['J2'])} switches, sim_steps={sim_steps}")

    bins = np.arange(0, max_sim_steps + BIN_SIZE, BIN_SIZE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"Switch timing (Q-learning) — {RUN_LABEL}\n{config_str}\ngenerated {timestamp}",
        fontsize=10,
    )

    avg_interval_overall = {}
    for ax, tl in zip(axes, TL_IDS):
        steps = all_switch_steps[tl]
        ax.hist(steps, bins=bins, color="steelblue", edgecolor="white")
        ax.set_title(f"{tl}: switch timing across episode (n={len(steps)} "
                      f"switches over {N_EPISODES} episodes)")
        ax.set_ylabel("switch count per 100-step bin")
        if steps:
            counts, _ = np.histogram(steps, bins=bins)
            peak_bin = bins[np.argmax(counts)]
            avg_interval = mean_switch_interval(steps)
            avg_interval_overall[tl] = avg_interval
            ax.axvline(peak_bin, color="red", linestyle="--", linewidth=1,
                       label=f"peak bin: step {peak_bin}-{peak_bin+BIN_SIZE}")
            ax.axvline(0, color="none")  # keep axis anchored at 0
            ax.text(0.98, 0.92, f"avg switch interval ≈ {avg_interval:.1f} steps",
                     transform=ax.transAxes, ha="right", va="top", fontsize=9,
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            ax.legend()
        else:
            avg_interval_overall[tl] = float("nan")

    axes[-1].set_xlabel("simulation step within episode")
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out_filename = f"switch_timing_plot_qlearning_{RUN_LABEL}_{timestamp}.png"
    out_path = os.path.join(SCRIPT_DIR, out_filename)
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot -> {out_path}")

    print(f"\nConfig: {config_str}")
    for tl in TL_IDS:
        steps = all_switch_steps[tl]
        counts, _ = np.histogram(steps, bins=bins)
        cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else float("nan")
        print(f"{tl}: coefficient of variation across time bins = {cv:.3f}")
        print(f"{tl}: mean switch interval (pooled over {N_EPISODES} episodes) "
              f"= {avg_interval_overall[tl]:.1f} sim steps")
        print(f"{tl}: mean switch interval per episode = "
              f"{np.nanmean(per_episode_intervals[tl]):.1f} sim steps "
              f"(± {np.nanstd(per_episode_intervals[tl]):.1f})")


if __name__ == "__main__":
    main()