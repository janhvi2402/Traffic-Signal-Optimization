import json
import glob
import matplotlib.pyplot as plt
import os

# ================================================================
# LOAD DATA
# ================================================================

with open("baseline_result.json") as f:
    baseline = json.load(f)

BASELINE_AVG = baseline["avg_wait_per_step"]

files = glob.glob("result_*.json")
results = []
for fp in files:
    with open(fp) as f:
        results.append(json.load(f))

print(f"Baseline avg wait : {BASELINE_AVG:.2f}s")
print(f"Experiments loaded: {len(results)}")

os.makedirs("plots", exist_ok=True)

# NEW: defaults for the three params added by the relative-action /
# min-green-floor / reward-shaping changes to train.py. Every fixed={...}
# filter below now pins these explicitly -- without pinning them, runs with
# different switch_penalty/wrong_direction_penalty/min_green_cycles would
# get silently lumped together into what looks like a clean alpha/gamma/etc
# sensitivity curve, when really the penalty settings are also varying.
# If your sweep harness uses different defaults than train.py's, change
# these three values to match.
DEFAULT_MIN_GREEN_CYCLES        = 1
DEFAULT_SWITCH_PENALTY          = 0.5
DEFAULT_WRONG_DIRECTION_PENALTY = 1.0


# ================================================================
# HELPERS
# ================================================================

def improvement(avg):
    return ((BASELINE_AVG - avg) / BASELINE_AVG) * 100

def filter_results(fixed: dict):
    out = []
    for r in results:
        match = True
        for k, v in fixed.items():
            if k not in r:
                # NEW: older result_*.json files (from before the reward-shaping
                # changes) won't have these keys at all -- skip them here rather
                # than crash, so old and new sweep files don't get silently mixed.
                match = False
                continue
            if isinstance(v, float):
                if abs(r[k] - v) > 1e-9:
                    match = False
            else:
                if r[k] != v:
                    match = False
        if match:
            out.append(r)
    return out

def single_plot(vary_key, vary_label, fixed, color, filename, title):
    data = filter_results(fixed)
    data.sort(key=lambda x: x[vary_key])
    if len(data) < 2:
        print(f"Skipping {filename} — need at least 2 data points")
        return
    x = [r[vary_key] for r in data]
    y = [r["avg_wait_per_step"] for r in data]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, marker="o", color=color, linewidth=2, markersize=7, label="RL Agent")
    ax.axhline(BASELINE_AVG, color="red", linestyle="--", linewidth=2,
               label=f"Baseline ({BASELINE_AVG:.2f}s)")
    for xi, yi in zip(x, y):
        ax.annotate(f"{improvement(yi):+.1f}%", (xi, yi),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)
    ax.set_xlabel(vary_label)
    ax.set_ylabel("Avg Wait per Step (s)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"plots/{filename}.png", dpi=150)
    plt.close()
    print(f"Saved plots/{filename}.png")


def joint_plot(vary_key1, vary_key2, label1, label2, fixed, filename, title):
    data = filter_results(fixed)
    if len(data) < 2:
        print(f"Skipping {filename} — need at least 2 data points")
        return
    groups = {}
    for r in data:
        k = r[vary_key2]
        groups.setdefault(k, []).append(r)

    colors = ["steelblue", "green", "orange", "purple"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (k2_val, group) in enumerate(sorted(groups.items())):
        group.sort(key=lambda x: x[vary_key1])
        ax.plot([r[vary_key1] for r in group],
                [r["avg_wait_per_step"] for r in group],
                marker="o", linewidth=2,
                color=colors[i % len(colors)],
                label=f"{label2}={k2_val}")
    ax.axhline(BASELINE_AVG, color="red", linestyle="--", linewidth=2,
               label=f"Baseline ({BASELINE_AVG:.2f}s)")
    ax.set_xlabel(label1)
    ax.set_ylabel("Avg Wait per Step (s)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"plots/{filename}.png", dpi=150)
    plt.close()
    print(f"Saved plots/{filename}.png")


# ================================================================
# INDIVIDUAL SENSITIVITY PLOTS
# NEW: min_green_cycles / switch_penalty / wrong_direction_penalty pinned
# in every fixed={} filter below so these plots aren't contaminated by
# un-pinned variation in the new reward-shaping params.
# ================================================================

COMMON_FIXED = {
    "green_time": 10,
    "yellow_time": 3,
    "epsilon_decay": 0.98,
    "min_green_cycles": DEFAULT_MIN_GREEN_CYCLES,
    "switch_penalty": DEFAULT_SWITCH_PENALTY,
    "wrong_direction_penalty": DEFAULT_WRONG_DIRECTION_PENALTY,
}

single_plot("alpha", "Alpha (Learning Rate)",
    fixed={**COMMON_FIXED, "gamma": 0.95, "episodes": 150},
    color="steelblue", filename="01_alpha", title="Alpha Sensitivity")

single_plot("gamma", "Gamma (Discount Factor)",
    fixed={**COMMON_FIXED, "alpha": 0.1, "episodes": 150},
    color="green", filename="02_gamma", title="Gamma Sensitivity")

single_plot("episodes", "Training Episodes",
    fixed={**COMMON_FIXED, "alpha": 0.1, "gamma": 0.95},
    color="orange", filename="03_episodes", title="Episodes Sensitivity")

single_plot("green_time", "Green Time (steps)",
    fixed={"alpha": 0.1, "gamma": 0.95, "episodes": 150, "yellow_time": 3,
           "epsilon_decay": 0.98, "min_green_cycles": DEFAULT_MIN_GREEN_CYCLES,
           "switch_penalty": DEFAULT_SWITCH_PENALTY,
           "wrong_direction_penalty": DEFAULT_WRONG_DIRECTION_PENALTY},
    color="purple", filename="04_green_time", title="Green Time Sensitivity")

single_plot("yellow_time", "Yellow Time (steps)",
    fixed={"alpha": 0.1, "gamma": 0.95, "episodes": 150, "green_time": 10,
           "epsilon_decay": 0.98, "min_green_cycles": DEFAULT_MIN_GREEN_CYCLES,
           "switch_penalty": DEFAULT_SWITCH_PENALTY,
           "wrong_direction_penalty": DEFAULT_WRONG_DIRECTION_PENALTY},
    color="brown", filename="05_yellow_time", title="Yellow Time Sensitivity")

single_plot("epsilon_decay", "Epsilon Decay",
    fixed={"alpha": 0.1, "gamma": 0.95, "episodes": 150, "green_time": 10,
           "yellow_time": 3, "min_green_cycles": DEFAULT_MIN_GREEN_CYCLES,
           "switch_penalty": DEFAULT_SWITCH_PENALTY,
           "wrong_direction_penalty": DEFAULT_WRONG_DIRECTION_PENALTY},
    color="teal", filename="06_decay", title="Epsilon Decay Sensitivity")

# NEW: sensitivity plots for the three added reward/env params
single_plot("min_green_cycles", "Min Green (cycles)",
    fixed={"alpha": 0.1, "gamma": 0.95, "episodes": 150, "green_time": 10,
           "yellow_time": 3, "epsilon_decay": 0.98,
           "switch_penalty": DEFAULT_SWITCH_PENALTY,
           "wrong_direction_penalty": DEFAULT_WRONG_DIRECTION_PENALTY},
    color="crimson", filename="07_min_green_cycles", title="Min Green Cycles Sensitivity")

single_plot("switch_penalty", "Switch Penalty",
    fixed={"alpha": 0.1, "gamma": 0.95, "episodes": 150, "green_time": 10,
           "yellow_time": 3, "epsilon_decay": 0.98,
           "min_green_cycles": DEFAULT_MIN_GREEN_CYCLES,
           "wrong_direction_penalty": DEFAULT_WRONG_DIRECTION_PENALTY},
    color="darkcyan", filename="08_switch_penalty", title="Switch Penalty Sensitivity")

single_plot("wrong_direction_penalty", "Wrong-Direction Penalty",
    fixed={"alpha": 0.1, "gamma": 0.95, "episodes": 150, "green_time": 10,
           "yellow_time": 3, "epsilon_decay": 0.98,
           "min_green_cycles": DEFAULT_MIN_GREEN_CYCLES,
           "switch_penalty": DEFAULT_SWITCH_PENALTY},
    color="darkgoldenrod", filename="09_wrong_direction_penalty",
    title="Wrong-Direction Penalty Sensitivity")


# ================================================================
# JOINT SENSITIVITY PLOTS
# ================================================================

joint_plot("alpha", "gamma", "Alpha", "Gamma",
    fixed={**COMMON_FIXED, "episodes": 150},
    filename="10_alpha_gamma", title="Alpha + Gamma Joint")

joint_plot("gamma", "episodes", "Gamma", "Episodes",
    fixed={**COMMON_FIXED, "alpha": 0.1},
    filename="11_gamma_episodes", title="Gamma + Episodes Joint")

joint_plot("alpha", "episodes", "Alpha", "Episodes",
    fixed={**COMMON_FIXED, "gamma": 0.95},
    filename="12_alpha_episodes", title="Alpha + Episodes Joint")

# NEW: joint plot for the two reward-shaping penalties -- useful to see
# whether they trade off against each other (e.g. does a low switch_penalty
# need a higher wrong_direction_penalty to compensate, or vice versa)
joint_plot("switch_penalty", "wrong_direction_penalty",
    "Switch Penalty", "Wrong-Direction Penalty",
    fixed={"alpha": 0.1, "gamma": 0.95, "episodes": 150, "green_time": 10,
           "yellow_time": 3, "epsilon_decay": 0.98,
           "min_green_cycles": DEFAULT_MIN_GREEN_CYCLES},
    filename="13_switch_wrongdir_joint", title="Switch Penalty + Wrong-Direction Penalty Joint")


# ================================================================
# OVERALL BAR CHART
# ================================================================

all_sorted = sorted(results, key=lambda x: x["avg_wait_per_step"])
labels     = [
    f"α={r['alpha']} γ={r['gamma']} ep={r['episodes']}\n"
    f"gt={r['green_time']} d={r['epsilon_decay']}\n"
    f"mg={r.get('min_green_cycles', '?')} sp={r.get('switch_penalty', '?')} "
    f"wp={r.get('wrong_direction_penalty', '?')}"
    for r in all_sorted
]
avg_waits  = [r["avg_wait_per_step"] for r in all_sorted]
bar_colors = ["green" if w < BASELINE_AVG else "steelblue" for w in avg_waits]

fig, ax = plt.subplots(figsize=(max(14, len(all_sorted)*1.2), 6))
bars = ax.bar(labels, avg_waits, color=bar_colors)
ax.axhline(BASELINE_AVG, color="red", linestyle="--", linewidth=2,
           label=f"Baseline ({BASELINE_AVG:.2f}s)")
for bar, w in zip(bars, avg_waits):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f"{improvement(w):+.1f}%",
            ha="center", va="bottom", fontsize=7)
ax.set_ylabel("Avg Wait per Step (s)")
ax.set_title("All Experiments vs Baseline (green = beats baseline)")
ax.legend()
ax.tick_params(axis="x", labelsize=6)
fig.tight_layout()
fig.savefig("plots/14_overall.png", dpi=150)
plt.close()
print("Saved plots/14_overall.png")


# ================================================================
# SUMMARY TABLE
# ================================================================

lines = []
lines.append("=" * 90)
lines.append("EXPERIMENT SUMMARY")
lines.append("=" * 90)
lines.append(f"{'Config':<70} {'Avg Wait':>10} {'Improvement':>12}")
lines.append("-" * 90)
lines.append(f"{'Baseline (Fixed Timing)':<70} {BASELINE_AVG:>10.2f} {'—':>12}")

for r in all_sorted:
    label = (f"scenario={r.get('scenario','medium')} "
         f"a={r['alpha']} g={r['gamma']} ep={r['episodes']} "
         f"mg={r.get('min_green_cycles', '?')} "
         f"sp={r.get('switch_penalty', '?')} wp={r.get('wrong_direction_penalty', '?')}")
    pct = improvement(r["avg_wait_per_step"])
    lines.append(f"{label:<70} {r['avg_wait_per_step']:>10.2f} {pct:>+11.1f}%")

lines.append("=" * 90)
best = all_sorted[0]
lines.append(f"\nBEST CONFIG:")
lines.append(f"  alpha={best['alpha']}, gamma={best['gamma']}, episodes={best['episodes']}")
lines.append(f"  green={best['green_time']}, yellow={best['yellow_time']}, decay={best['epsilon_decay']}")
lines.append(f"  min_green_cycles={best.get('min_green_cycles', '?')}, "
             f"switch_penalty={best.get('switch_penalty', '?')}, "
             f"wrong_direction_penalty={best.get('wrong_direction_penalty', '?')}")
lines.append(f"  Avg Wait   : {best['avg_wait_per_step']:.2f}s")
lines.append(f"  Improvement: {improvement(best['avg_wait_per_step']):+.1f}% over baseline")

summary = "\n".join(lines)
print("\n" + summary)

with open("plots/summary.txt", "w") as f:
    f.write(summary)

print("\nSaved plots/summary.txt")
print("All done — check the plots/ folder")