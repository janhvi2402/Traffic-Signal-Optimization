"""
plot_switch_timing_qlearning.py
==============================================================
Q-learning counterpart to plot_switch_timing.py (the PPO version).

Runs several evaluation episodes using the trained qtable.pkl
(same greedy-action logic as test.py), and digs into *why* switches
happen when they do:

  1. Real hold duration per switch (steps between consecutive
     switches of the same junction, computed WITHIN each episode --
     never pooled across an episode boundary, since step counters
     reset to 0 each episode).
  2. Flip-flop rate: % of switches at the theoretical minimum
     interval (YELLOW_TIME + GREEN_TIME) -- i.e. switched back on the
     very next decision.
  3. Premature-switch rate: at the moment of each switch, was the arm
     LOSING green still more congested than the arm GAINING green?
  4. NEW -- does hold length track queue density? For every hold,
     pairs the queue that was WAITING when the phase started (i.e.
     the backlog the junction just switched onto) with how long that
     phase was then held before switching away again. Buckets by that
     starting backlog (same bucket() thresholds as training) and
     reports mean hold duration per bucket, plus the Pearson
     correlation between starting backlog and hold length. If a
     learned, queue-responsive policy exists, hold duration should
     rise with starting backlog; if it's flat, the agent isn't
     actually using queue size to decide how long to hold.

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
import train as trainmod

N_EPISODES = 5
BIN_SIZE   = 100
TL_IDS     = ["J1", "J2"]
MIN_INTERVAL = trainmod.YELLOW_TIME + trainmod.GREEN_TIME   # fastest possible switch-back
N_BUCKETS = 7   # bucket() returns 0..6

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
    arm of a junction, given its phase."""
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

    all_switch_steps   = {tl: [] for tl in TL_IDS}
    all_hold_durations = {tl: [] for tl in TL_IDS}
    all_events         = {tl: [] for tl in TL_IDS}
    # NEW: (initial_backlog, hold_duration, final_backlog) triples, paired
    # WITHIN each episode only -- initial_backlog is the red_q logged at the
    # PREVIOUS switch (the queue this phase inherited when it turned green).
    hold_vs_backlog = {tl: [] for tl in TL_IDS}
    max_sim_steps = 0

    for ep in range(N_EPISODES):
        switch_events, sim_steps = run_episode(q_table, seed=ep)
        max_sim_steps = max(max_sim_steps, sim_steps)
        for tl in TL_IDS:
            events = switch_events[tl]
            all_switch_steps[tl].extend(e["step"] for e in events)
            all_events[tl].extend(events)

            steps_this_ep = [e["step"] for e in events]
            if len(steps_this_ep) >= 2:
                all_hold_durations[tl].extend(np.diff(steps_this_ep).tolist())

            for i in range(1, len(events)):
                initial_backlog = events[i - 1]["red_q"]   # queue this phase inherited
                hold_duration   = events[i]["step"] - events[i - 1]["step"]
                final_backlog   = events[i]["green_q"]     # residual queue when it ended
                hold_vs_backlog[tl].append((initial_backlog, hold_duration, final_backlog))

        print(f"episode {ep} done — J1: {len(switch_events['J1'])} switches, "
              f"J2: {len(switch_events['J2'])} switches, sim_steps={sim_steps}")

    bins = np.arange(0, max_sim_steps + BIN_SIZE, BIN_SIZE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    fig, axes = plt.subplots(4, 2, figsize=(14, 17))
    fig.suptitle(
        f"Switch timing & policy sanity (Q-learning) — {RUN_LABEL}\n{config_str}\ngenerated {timestamp}",
        fontsize=10,
    )

    insights = []

    for col, tl in enumerate(TL_IDS):
        steps = all_switch_steps[tl]
        holds = np.array(all_hold_durations[tl])
        events = all_events[tl]
        triples = hold_vs_backlog[tl]

        # (row 0) timing histogram
        ax = axes[0, col]
        ax.hist(steps, bins=bins, color="steelblue", edgecolor="white")
        ax.set_title(f"{tl}: switch timing (n={len(steps)} switches / {N_EPISODES} eps)")
        ax.set_xlabel("simulation step within episode")
        ax.set_ylabel("switch count per 100-step bin")

        # (row 1) hold-duration histogram
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

        # (row 2) NEW -- mean hold duration by starting-backlog bucket
        ax = axes[2, col]
        if len(triples) > 0:
            init_q  = np.array([t[0] for t in triples], dtype=float)
            hold_d  = np.array([t[1] for t in triples], dtype=float)
            buckets = np.array([bucket(q) for q in init_q])

            means, stds, counts = [], [], []
            for b in range(N_BUCKETS):
                mask = buckets == b
                if mask.sum() > 0:
                    means.append(hold_d[mask].mean())
                    stds.append(hold_d[mask].std())
                    counts.append(int(mask.sum()))
                else:
                    means.append(np.nan)
                    stds.append(0)
                    counts.append(0)

            x = np.arange(N_BUCKETS)
            bars = ax.bar(x, means, yerr=stds, color="seagreen", capsize=3, edgecolor="black")
            ax.axhline(MIN_INTERVAL, color="crimson", linestyle="--", linewidth=1,
                       label=f"min possible = {MIN_INTERVAL}")
            for xi, m, c in zip(x, means, counts):
                if not np.isnan(m):
                    ax.text(xi, m + (stds[x.tolist().index(xi)] if not np.isnan(stds[x.tolist().index(xi)]) else 0) + 1,
                             f"n={c}", ha="center", fontsize=7)
            ax.set_xticks(x)
            ax.set_xlabel("starting-backlog bucket (queue this phase inherited)\n0=empty ... 6=25+ vehicles")
            ax.set_ylabel("mean hold duration (steps)")
            ax.legend(fontsize=8)

            # Pearson correlation between raw starting backlog and hold length
            if np.std(init_q) > 0 and np.std(hold_d) > 0:
                corr = float(np.corrcoef(init_q, hold_d)[0, 1])
            else:
                corr = float("nan")
        else:
            corr = float("nan")
        ax.set_title(f"{tl}: does hold length track starting backlog?")

        # (row 3) summary text panel
        ax = axes[3, col]
        ax.axis("off")
        if len(holds) > 0 and len(events) > 0:
            flip_flop_rate = 100 * np.mean(holds == MIN_INTERVAL)
            near_min_rate  = 100 * np.mean(holds <= MIN_INTERVAL + trainmod.GREEN_TIME)
            premature = np.array([e["green_q"] > e["red_q"] for e in events])
            premature_rate = 100 * premature.mean()
            mean_green_q = np.mean([e["green_q"] for e in events])
            mean_red_q   = np.mean([e["red_q"] for e in events])
            mean_hold_green_s = holds.mean() - trainmod.YELLOW_TIME

            summary = (
                f"Mean hold duration : {holds.mean():.1f} steps  (median {np.median(holds):.0f})\n"
                f"  -> ~{mean_hold_green_s:.1f}s of actual GREEN time before switching away\n"
                f"Min possible hold  : {MIN_INTERVAL} steps\n"
                f"Flip-flop rate     : {flip_flop_rate:.1f}%  (switches back at min interval)\n"
                f"Near-min hold rate : {near_min_rate:.1f}%  (<= min + 1 cycle)\n\n"
                f"Mean queue LOSING green at switch : {mean_green_q:.1f}\n"
                f"Mean queue GAINING green at switch: {mean_red_q:.1f}\n"
                f"Premature-switch rate : {premature_rate:.1f}%\n\n"
                f"Corr(starting backlog, hold length): r = {corr:.2f}\n"
                f"  (near 0 = hold length is NOT queue-responsive;\n"
                f"   positive = holds longer when it inherits a bigger queue)"
            )
            ax.text(0.02, 0.95, summary, va="top", ha="left", fontsize=9.5, family="monospace")

            insights.append(
                f"{tl}: mean hold {holds.mean():.1f} steps (~{mean_hold_green_s:.1f}s green), "
                f"flip-flop rate {flip_flop_rate:.1f}%, premature-switch rate {premature_rate:.1f}%, "
                f"corr(backlog, hold)={corr:.2f}."
            )
            if flip_flop_rate > 30:
                insights.append(
                    f"  WARNING ({tl}): >{flip_flop_rate:.0f}% of switches happen at the minimum "
                    f"possible interval -- policy looks like it's toggling on a fixed rhythm "
                    f"rather than holding green while a queue clears."
                )
            if not np.isnan(corr) and abs(corr) < 0.15:
                insights.append(
                    f"  WARNING ({tl}): correlation between starting backlog and hold length is "
                    f"essentially flat (r={corr:.2f}) -- hold duration does not appear to be "
                    f"driven by how congested the arm was when it got the green."
                )
        else:
            ax.text(0.5, 0.5, "not enough switch events", ha="center", va="center")

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_filename = f"switch_timing_plot_qlearning_{RUN_LABEL}_{timestamp}.png"
    out_path = os.path.join(SCRIPT_DIR, out_filename)
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot -> {out_path}")

    print(f"\nConfig: {config_str}\n")
    print("\n".join(insights))


if __name__ == "__main__":
    main()