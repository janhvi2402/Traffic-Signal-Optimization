"""
plot_switch_timing_qlearning.py
==============================================================
Q-learning counterpart to plot_switch_timing.py (the PPO version).

Runs several evaluation episodes using the trained qtable.pkl, with the
SAME relative-action / min-green-floor / hold-duration-state logic as the
updated train.py and test.py (must match exactly, or the Q-table lookups
here won't mean the same thing they meant during training).

Digs into *why* switches happen when they do:

  1. Real hold duration per switch (steps between consecutive switches of
     the same junction, computed WITHIN each episode only).
  2. Flip-flop rate: % of switches at the enforced minimum interval
     (MIN_GREEN_CYCLES * GREEN_TIME + YELLOW_TIME) -- i.e. switched back
     at the earliest cycle the min-green floor allowed.
  3. Premature-switch rate: at the moment of each switch, was the arm
     LOSING green still more congested than the arm GAINING green?
  4. Does hold length track queue density? Pairs the queue that was
     waiting when the phase started (the backlog it just switched onto)
     with how long that phase was then held before switching away again.
     Buckets by that starting backlog and reports mean hold duration per
     bucket, plus the Pearson correlation between starting backlog and
     hold length.

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
#
# IMPORTANT: this imports a module literally named "train" (train.py in
# this same folder). If your training script is actually named
# something else (e.g. model.py), change the import below to match --
# otherwise this will raise ModuleNotFoundError, or worse, silently
# import a stale/different file of the same name from elsewhere on
# sys.path.
import model as trainmod

N_EPISODES = 5
BIN_SIZE   = 100
TL_IDS     = ["J1", "J2"]
MIN_INTERVAL = trainmod.MIN_GREEN_CYCLES * trainmod.GREEN_TIME + trainmod.YELLOW_TIME
# CHANGED: 7 -> 8, must match trainmod.bucket()'s return range exactly
# (0..7 after the resolution fix). Leaving this at 7 silently drops every
# switch whose starting backlog fell in bucket 7 (the biggest-queue
# bucket) from the "hold length vs backlog" panel instead of erroring --
# exactly the switches most relevant to the duration-sensitivity question.
N_BUCKETS = 8   # bucket() returns 0..7

RUN_LABEL = "qlearning_v3_finer_buckets_5s_cycle"   # <-- update this each time you retrain


def run_episode(q_table, seed):
    """Same control loop as the updated test.py's run_qlearning_episode, but
    records, for every ACTUAL switch (i.e. after the min-green mask), the
    sim step and the green/red arm queue counts at the moment it was decided."""
    traci.start(["sumo", "-c", SUMOCFG_PATH, "--no-warnings", "--seed", str(seed)])
    traci.simulationStep()

    j1_phase = 0
    j2_phase = 0
    j1_hold  = 0
    j2_hold  = 0
    traci.trafficlight.setPhase(trainmod.J1, j1_phase)
    traci.trafficlight.setPhaseDuration(trainmod.J1, 9999)
    traci.trafficlight.setPhase(trainmod.J2, j2_phase)
    traci.trafficlight.setPhaseDuration(trainmod.J2, 9999)

    sim_steps = 0
    switch_events = {"J1": [], "J2": []}   # list of dicts: step, green_q, red_q

    while traci.simulation.getMinExpectedNumber() > 0:
        state = trainmod.get_state(j1_phase, j2_phase, j1_hold, j2_hold)
        if state in q_table:
            action_idx = int(np.argmax(q_table[state]))
        else:
            action_idx = 0

        j1_choice, j2_choice = trainmod.ACTION_SPACE[action_idx]
        j1_switching = bool(j1_choice) and (j1_hold >= trainmod.MIN_GREEN_CYCLES)
        j2_switching = bool(j2_choice) and (j2_hold >= trainmod.MIN_GREEN_CYCLES)

        if j1_switching:
            g, r = trainmod.get_arm_queues("J1", j1_phase)
            switch_events["J1"].append({"step": sim_steps, "green_q": g, "red_q": r})
        if j2_switching:
            g, r = trainmod.get_arm_queues("J2", j2_phase)
            switch_events["J2"].append({"step": sim_steps, "green_q": g, "red_q": r})

        j1_new = (2 - j1_phase) if j1_switching else j1_phase
        j2_new = (2 - j2_phase) if j2_switching else j2_phase

        if j1_switching:
            traci.trafficlight.setPhase(trainmod.J1, trainmod.YELLOW_PHASE[j1_phase])
            traci.trafficlight.setPhaseDuration(trainmod.J1, 9999)
        if j2_switching:
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
        j1_hold = 0 if j1_switching else j1_hold + 1
        j2_hold = 0 if j2_switching else j2_hold + 1

        for _ in range(trainmod.GREEN_TIME):
            traci.simulationStep()
            sim_steps += 1

    traci.close()
    return switch_events, sim_steps


def bucket(x):
    return trainmod.bucket(x)


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
        f"yellow_time={trainmod.YELLOW_TIME}, min_green_cycles={trainmod.MIN_GREEN_CYCLES}, "
        f"switch_penalty={trainmod.SWITCH_PENALTY}, "
        f"wrong_direction_penalty={trainmod.WRONG_DIRECTION_PENALTY}\n"
        f"states_learned={len(q_table)}, min_possible_interval={MIN_INTERVAL} steps"
    )

    all_switch_steps   = {tl: [] for tl in TL_IDS}
    all_hold_durations = {tl: [] for tl in TL_IDS}
    all_events         = {tl: [] for tl in TL_IDS}
    hold_vs_backlog    = {tl: [] for tl in TL_IDS}
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
                initial_backlog = events[i - 1]["red_q"]
                hold_duration   = events[i]["step"] - events[i - 1]["step"]
                final_backlog   = events[i]["green_q"]
                hold_vs_backlog[tl].append((initial_backlog, hold_duration, final_backlog))

        print(f"episode {ep} done — J1: {len(switch_events['J1'])} switches, "
              f"J2: {len(switch_events['J2'])} switches, sim_steps={sim_steps}")

    bins = np.arange(0, max_sim_steps + BIN_SIZE, BIN_SIZE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    fig, axes = plt.subplots(4, 2, figsize=(14, 17))
    fig.suptitle(
        f"Switch timing & policy sanity (Q-learning) — {RUN_LABEL}\n{config_str}\ngenerated {timestamp}",
        fontsize=9,
    )

    insights = []

    for col, tl in enumerate(TL_IDS):
        steps = all_switch_steps[tl]
        holds = np.array(all_hold_durations[tl])
        events = all_events[tl]
        triples = hold_vs_backlog[tl]

        ax = axes[0, col]
        ax.hist(steps, bins=bins, color="steelblue", edgecolor="white")
        ax.set_title(f"{tl}: switch timing (n={len(steps)} switches / {N_EPISODES} eps)")
        ax.set_xlabel("simulation step within episode")
        ax.set_ylabel("switch count per 100-step bin")

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

        ax = axes[2, col]
        corr = float("nan")
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
            ax.bar(x, means, yerr=stds, color="seagreen", capsize=3, edgecolor="black")
            ax.axhline(MIN_INTERVAL, color="crimson", linestyle="--", linewidth=1,
                       label=f"min possible = {MIN_INTERVAL}")
            for xi, m, s, c in zip(x, means, stds, counts):
                if not np.isnan(m):
                    ax.text(xi, m + (s if not np.isnan(s) else 0) + 1, f"n={c}", ha="center", fontsize=7)
            ax.set_xticks(x)
            ax.set_xlabel("starting-backlog bucket (queue this phase inherited)\n0=empty ... 7=21+ vehicles")
            ax.set_ylabel("mean hold duration (steps)")
            ax.legend(fontsize=8)

            if np.std(init_q) > 0 and np.std(hold_d) > 0:
                corr = float(np.corrcoef(init_q, hold_d)[0, 1])
        ax.set_title(f"{tl}: does hold length track starting backlog?")

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
                f"Min possible hold  : {MIN_INTERVAL} steps  (min_green_cycles={trainmod.MIN_GREEN_CYCLES})\n"
                f"Flip-flop rate     : {flip_flop_rate:.1f}%  (switches at min interval)\n"
                f"Near-min hold rate : {near_min_rate:.1f}%  (<= min + 1 cycle)\n\n"
                f"Mean queue LOSING green at switch : {mean_green_q:.1f}\n"
                f"Mean queue GAINING green at switch: {mean_red_q:.1f}\n"
                f"Premature-switch rate : {premature_rate:.1f}%\n\n"
                f"Corr(starting backlog, hold length): r = {corr:.2f}\n"
                f"  (near 0 = still not queue-responsive;\n"
                f"   positive = holds longer for bigger inherited queue)"
            )
            ax.text(0.02, 0.95, summary, va="top", ha="left", fontsize=9, family="monospace")

            insights.append(
                f"{tl}: mean hold {holds.mean():.1f} steps (~{mean_hold_green_s:.1f}s green), "
                f"flip-flop rate {flip_flop_rate:.1f}%, premature-switch rate {premature_rate:.1f}%, "
                f"corr(backlog, hold)={corr:.2f}."
            )
        else:
            ax.text(0.5, 0.5, "not enough switch events", ha="center", va="center")

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_filename = f"switch_timing_plot_qlearning_{RUN_LABEL}_{timestamp}.png"
    out_path = os.path.join(SCRIPT_DIR, out_filename)
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot -> {out_path}")

    print(f"\nConfig: {config_str}\n")
    print("\n".join(insights))


if __name__ == "__main__":
    main()