"""
diagnostic.py
==============================================================
Report-ready diagnostic plots for the tabular Q-learning
2-junction SUMO traffic signal controller.

Reads whichever of these exist (degrades gracefully otherwise):
  - qtable.pkl                              -> Q-table quality diagnostics
  - training_log.csv                        -> training convergence diagnostics
                                                (produced by the patched train.py)
  - results/qlearning_vs_fixed_unified.json -> test-time performance diagnostics
                                                (produced by the patched test.py)

Run from the Q-learning project root:
    python diagnostic.py

Outputs land in diagnostics/*.png plus diagnostics/insights.txt
(auto-written bullet points you can paste straight into your report).
==============================================================
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
QTABLE_PATH     = os.path.join(SCRIPT_DIR, "qtable.pkl")
TRAINLOG_PATH   = os.path.join(SCRIPT_DIR, "training_log.csv")
TESTRESULT_PATH = os.path.join(SCRIPT_DIR, "results", "qlearning_vs_fixed_unified.json")
OUT_DIR         = os.path.join(SCRIPT_DIR, "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)

ACTION_SPACE  = [(0, 0), (0, 2), (2, 0), (2, 2)]
ACTION_LABELS = ["Both NS\n(0,0)", "J1 NS / J2 EW\n(0,2)", "J1 EW / J2 NS\n(2,0)", "Both EW\n(2,2)"]

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

insights = []  # collects auto-written report bullets


# 1. Q-TABLE QUALITY DIAGNOSTICS

def run_qtable_diagnostics():
    if not os.path.exists(QTABLE_PATH):
        print(f"[skip] qtable.pkl not found at {QTABLE_PATH}")
        return

    with open(QTABLE_PATH, "rb") as f:
        q_table = pickle.load(f)

    states = list(q_table.keys())
    n_states = len(states)
    if n_states == 0:
        print("[skip] qtable.pkl is empty")
        return

    q_values = np.array([q_table[s] for s in states])       # (n_states, 4)
    best_actions = np.argmax(q_values, axis=1)
    q_spread = q_values.max(axis=1) - q_values.min(axis=1)   # "decision confidence"
    q_max = q_values.max(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Q-Table Quality Diagnostics", fontsize=14, fontweight="bold")

    # (a) greedy action preference across learned states
    ax = axes[0, 0]
    counts = np.bincount(best_actions, minlength=len(ACTION_SPACE))
    bars = ax.bar(ACTION_LABELS, counts, color="steelblue")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 f"{c}\n({100*c/n_states:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Greedy Action Preference Across {n_states} Learned States")
    ax.set_ylabel("Number of states")

    # (b) decision confidence distribution
    ax = axes[0, 1]
    ax.hist(q_spread, bins=30, color="seagreen", edgecolor="black")
    ax.axvline(np.median(q_spread), color="red", linestyle="--",
               label=f"median={np.median(q_spread):.2f}")
    ax.set_title("Decision Confidence (max Q − min Q per state)")
    ax.set_xlabel("Q-value spread")
    ax.set_ylabel("Number of states")
    ax.legend()

    # (c) best-Q-value distribution
    ax = axes[1, 0]
    ax.hist(q_max, bins=30, color="darkorange", edgecolor="black")
    ax.set_title("Distribution of Best Q-value per State")
    ax.set_xlabel("max Q(s, ·)")
    ax.set_ylabel("Number of states")

    # (d) policy sanity check: does the agent switch when it should?
    # state layout: (j1_green, j1_red, j2_green, j2_red, j1_phase_feat, j2_phase_feat)
    j1_green      = np.array([s[0] for s in states])
    j1_red        = np.array([s[1] for s in states])
    j1_phase_feat = np.array([s[4] for s in states])      # 0 -> currently phase 0, 1 -> currently phase 2
    current_j1_phase = j1_phase_feat * 2
    best_j1_new   = np.array([ACTION_SPACE[a][0] for a in best_actions])
    switch_j1     = (best_j1_new != current_j1_phase).astype(float)
    imbalance     = j1_green - j1_red   # positive => green arm more congested than red arm

    diffs = sorted(set(imbalance.tolist()))
    xs, ys, ns = [], [], []
    for d in diffs:
        mask = imbalance == d
        if mask.sum() >= 3:   # skip near-empty bins, too noisy to plot
            xs.append(d)
            ys.append(switch_j1[mask].mean())
            ns.append(int(mask.sum()))

    ax = axes[1, 1]
    ax.plot(xs, ys, marker="o", color="crimson", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle=":")
    ax.set_xlabel("Green-arm queue bucket − Red-arm queue bucket")
    ax.set_ylabel("P(agent switches J1 phase)")
    ax.set_title("Policy Sanity Check: Switch Probability vs Queue Imbalance\n"
                  "(should fall as the green arm gets more congested)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, "01_qtable_quality.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

    # ---- insights ----
    dominant_action_pct = 100 * counts.max() / n_states
    insights.append(f"Q-table covers {n_states} distinct states.")
    insights.append(
        f"Greedy policy is spread across all {len(ACTION_SPACE)} actions "
        f"(most-picked action accounts for {dominant_action_pct:.1f}% of states)"
        + (" — no single action dominates, so the policy is not degenerate."
           if dominant_action_pct < 70 else
           " — check whether this action is disproportionately favoured for a reason "
           "(e.g. dataset imbalance) or is a sign of shortcut/degenerate behaviour.")
    )
    insights.append(f"Median decision confidence (Q-value spread) is {np.median(q_spread):.2f}.")
    if len(xs) >= 2 and ys[0] < ys[-1]:
        insights.append(
            "WARNING: switch probability increases with green-arm congestion in the sanity-check "
            "plot — this is backwards from what a well-behaved policy should do. Investigate."
        )
    elif len(xs) >= 2:
        insights.append(
            "Switch-probability-vs-imbalance plot decreases as expected: the agent is less likely "
            "to switch away from a junction that is currently serving its congested arm, "
            "supporting that it is conditioning on queue state rather than a fixed timer."
        )


# 2. TRAINING CONVERGENCE DIAGNOSTICS

def run_training_diagnostics():
    if not os.path.exists(TRAINLOG_PATH):
        print(f"[skip] training_log.csv not found at {TRAINLOG_PATH} "
              f"(use the patched train.py, which writes this file, then re-run training)")
        return

    data = np.genfromtxt(TRAINLOG_PATH, delimiter=",", names=True)
    if data.size == 0:
        print("[skip] training_log.csv is empty")
        return

    episode      = data["episode"]
    steps        = data["steps"]
    total_reward = data["total_reward"]
    n_states     = data["n_states"]
    epsilon      = data["epsilon"]
    alpha        = data["alpha"]
    reward_per_step = total_reward / np.maximum(steps, 1)

    def rolling_mean(x, window=10):
        if len(x) < window:
            return x
        kernel = np.ones(window) / window
        return np.convolve(x, kernel, mode="valid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Training Convergence Diagnostics", fontsize=14, fontweight="bold")

    # (a) reward per episode with rolling mean
    ax = axes[0, 0]
    ax.plot(episode, total_reward, color="lightsteelblue", linewidth=1, label="Raw")
    rm = rolling_mean(total_reward)
    ax.plot(episode[-len(rm):], rm, color="navy", linewidth=2, label="10-ep rolling mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total episode reward")
    ax.set_title("Reward per Episode")
    ax.legend()

    # (b) reward per step (normalizes for variable episode length)
    ax = axes[0, 1]
    ax.plot(episode, reward_per_step, color="lightsalmon", linewidth=1, label="Raw")
    rm2 = rolling_mean(reward_per_step)
    ax.plot(episode[-len(rm2):], rm2, color="darkred", linewidth=2, label="10-ep rolling mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward / step")
    ax.set_title("Normalized Reward per Step")
    ax.legend()

    # (c) epsilon and alpha decay
    ax = axes[1, 0]
    ax.plot(episode, epsilon, color="teal", linewidth=2, label="epsilon (exploration)")
    ax.plot(episode, alpha, color="darkorange", linewidth=2, label="alpha (learning rate)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    ax.set_title("Exploration / Learning-rate Decay")
    ax.legend()

    # (d) Q-table growth (state discovery over training)
    ax = axes[1, 1]
    ax.plot(episode, n_states, color="seagreen", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative distinct states")
    ax.set_title("Q-table Growth (State-space Coverage over Training)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, "02_training_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

    # ---- insights ----
    first_half = reward_per_step[: len(reward_per_step) // 2]
    second_half = reward_per_step[len(reward_per_step) // 2:]
    insights.append(
        f"Mean reward/step improved from {first_half.mean():.3f} (first half of training) "
        f"to {second_half.mean():.3f} (second half)."
    )
    late_growth = n_states[-1] - n_states[len(n_states) // 2]
    insights.append(
        f"Q-table grew to {int(n_states[-1])} states by the end of training; "
        f"{int(late_growth)} of those were discovered in the second half "
        f"({'still expanding late — consider more episodes' if late_growth > 0.15 * n_states[-1] else 'growth largely plateaued, suggesting adequate coverage'})."
    )


# 3. TEST-TIME PERFORMANCE DIAGNOSTICS

def run_test_diagnostics():
    if not os.path.exists(TESTRESULT_PATH):
        print(f"[skip] {TESTRESULT_PATH} not found "
              f"(use the patched test.py, which saves per-episode arrays, then re-run eval)")
        return

    with open(TESTRESULT_PATH) as f:
        r = json.load(f)

    fixed_waits = r.get("fixed_waits_per_episode")
    ql_waits    = r.get("ql_waits_per_episode")
    coverage    = r.get("coverage_per_episode")

    if not fixed_waits or not ql_waits:
        print("[skip] qlearning_vs_fixed_unified.json has no per-episode arrays "
              "(use the patched test.py to generate them)")
        return

    fixed_waits = np.array(fixed_waits)
    ql_waits    = np.array(ql_waits)
    ep_improvement = (fixed_waits - ql_waits) / fixed_waits * 100

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Test-Time Performance Diagnostics", fontsize=14, fontweight="bold")

    # (a) fixed vs Q-learning per episode
    ax = axes[0, 0]
    x = np.arange(len(fixed_waits))
    width = 0.35
    ax.bar(x - width/2, fixed_waits, width, label="Fixed-time", color="lightcoral")
    ax.bar(x + width/2, ql_waits, width, label="Q-learning", color="steelblue")
    ax.set_xlabel("Test episode (seed)")
    ax.set_ylabel("Avg wait per step (s)")
    ax.set_title("Fixed-time vs Q-learning, per Test Episode")
    ax.set_xticks(x)
    ax.legend()

    # (b) improvement % per episode
    ax = axes[0, 1]
    colors = ["seagreen" if v > 0 else "crimson" for v in ep_improvement]
    ax.bar(x, ep_improvement, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Test episode (seed)")
    ax.set_ylabel("Improvement over fixed-time (%)")
    ax.set_title("Per-episode Improvement")
    ax.set_xticks(x)

    # (c) state coverage per episode
    ax = axes[1, 0]
    if coverage:
        ax.bar(x, np.array(coverage) * 100, color="darkorange")
        ax.set_ylim(0, 100)
        ax.set_ylabel("State coverage (%)")
        ax.set_xlabel("Test episode (seed)")
        ax.set_title("Fraction of Decisions Matched to a Known State")
        ax.set_xticks(x)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "coverage data not available", ha="center", va="center")

    # (d) summary text panel
    ax = axes[1, 1]
    ax.axis("off")
    mean_imp = ep_improvement.mean()
    summary_text = (
        f"Fixed-time mean wait : {fixed_waits.mean():.2f}s (± {fixed_waits.std():.2f})\n"
        f"Q-learning mean wait : {ql_waits.mean():.2f}s (± {ql_waits.std():.2f})\n\n"
        f"Mean improvement     : {mean_imp:+.1f}%\n"
        f"Best episode         : {ep_improvement.max():+.1f}%\n"
        f"Worst episode        : {ep_improvement.min():+.1f}%\n"
    )
    if coverage:
        summary_text += f"\nMean state coverage  : {100*np.mean(coverage):.1f}%"
    ax.text(0.05, 0.95, summary_text, va="top", ha="left", fontsize=11, family="monospace")
    ax.set_title("Summary")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, "03_test_performance.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

    # ---- insights ----
    insights.append(
        f"Across {len(fixed_waits)} test episodes, Q-learning improved average wait time by "
        f"{mean_imp:+.1f}% over fixed-time (range {ep_improvement.min():+.1f}% to {ep_improvement.max():+.1f}%)."
    )
    if coverage:
        insights.append(f"Mean test-time state coverage was {100*np.mean(coverage):.1f}%, "
                         f"i.e. the fraction of decisions where the visited state had been seen in training.")


# ==================================================================
# MAIN
# ==================================================================
def main():
    run_qtable_diagnostics()
    run_training_diagnostics()
    run_test_diagnostics()

    if insights:
        insights_path = os.path.join(OUT_DIR, "insights.txt")
        with open(insights_path, "w") as f:
            f.write("AUTO-GENERATED DIAGNOSTIC INSIGHTS\n")
            f.write("=" * 60 + "\n\n")
            for line in insights:
                f.write(f"- {line}\n")
        print(f"\nSaved {insights_path}")
        print("\n".join(f"- {line}" for line in insights))
    else:
        print("\nNo diagnostics could be run — none of qtable.pkl / training_log.csv / "
              "results/qlearning_vs_fixed_unified.json were found.")


if __name__ == "__main__":
    main()