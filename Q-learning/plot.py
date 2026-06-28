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
# ================================================================

single_plot("alpha", "Alpha (Learning Rate)",
    fixed={"gamma":0.95,"episodes":150,"green_time":10,"yellow_time":3,"epsilon_decay":0.98},
    color="steelblue", filename="01_alpha", title="Alpha Sensitivity")

single_plot("gamma", "Gamma (Discount Factor)",
    fixed={"alpha":0.1,"episodes":150,"green_time":10,"yellow_time":3,"epsilon_decay":0.98},
    color="green", filename="02_gamma", title="Gamma Sensitivity")

single_plot("episodes", "Training Episodes",
    fixed={"alpha":0.1,"gamma":0.95,"green_time":10,"yellow_time":3,"epsilon_decay":0.98},
    color="orange", filename="03_episodes", title="Episodes Sensitivity")

single_plot("green_time", "Green Time (steps)",
    fixed={"alpha":0.1,"gamma":0.95,"episodes":150,"yellow_time":3,"epsilon_decay":0.98},
    color="purple", filename="04_green_time", title="Green Time Sensitivity")

single_plot("yellow_time", "Yellow Time (steps)",
    fixed={"alpha":0.1,"gamma":0.95,"episodes":150,"green_time":10,"epsilon_decay":0.98},
    color="brown", filename="05_yellow_time", title="Yellow Time Sensitivity")

single_plot("epsilon_decay", "Epsilon Decay",
    fixed={"alpha":0.1,"gamma":0.95,"episodes":150,"green_time":10,"yellow_time":3},
    color="teal", filename="06_decay", title="Epsilon Decay Sensitivity")


# ================================================================
# JOINT SENSITIVITY PLOTS
# ================================================================

joint_plot("alpha", "gamma", "Alpha", "Gamma",
    fixed={"episodes":150,"green_time":10,"yellow_time":3,"epsilon_decay":0.98},
    filename="07_alpha_gamma", title="Alpha + Gamma Joint")

joint_plot("gamma", "episodes", "Gamma", "Episodes",
    fixed={"alpha":0.1,"green_time":10,"yellow_time":3,"epsilon_decay":0.98},
    filename="08_gamma_episodes", title="Gamma + Episodes Joint")

joint_plot("alpha", "episodes", "Alpha", "Episodes",
    fixed={"gamma":0.95,"green_time":10,"yellow_time":3,"epsilon_decay":0.98},
    filename="09_alpha_episodes", title="Alpha + Episodes Joint")


# ================================================================
# OVERALL BAR CHART
# ================================================================

all_sorted = sorted(results, key=lambda x: x["avg_wait_per_step"])
labels     = [
    f"α={r['alpha']} γ={r['gamma']} ep={r['episodes']}\ngt={r['green_time']} d={r['epsilon_decay']}"
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
ax.tick_params(axis="x", labelsize=7)
fig.tight_layout()
fig.savefig("plots/10_overall.png", dpi=150)
plt.close()
print("Saved plots/10_overall.png")


# ================================================================
# SUMMARY TABLE
# ================================================================

lines = []
lines.append("=" * 80)
lines.append("EXPERIMENT SUMMARY")
lines.append("=" * 80)
lines.append(f"{'Config':<50} {'Avg Wait':>10} {'Improvement':>12}")
lines.append("-" * 80)
lines.append(f"{'Baseline (Fixed Timing)':<50} {BASELINE_AVG:>10.2f} {'—':>12}")

for r in all_sorted:
    label = (f"scenario={r.get('scenario','medium')} "
         f"a={r['alpha']} g={r['gamma']} ep={r['episodes']}")
    pct = improvement(r["avg_wait_per_step"])
    lines.append(f"{label:<50} {r['avg_wait_per_step']:>10.2f} {pct:>+11.1f}%")

lines.append("=" * 80)
best = all_sorted[0]
lines.append(f"\nBEST CONFIG:")
lines.append(f"  alpha={best['alpha']}, gamma={best['gamma']}, episodes={best['episodes']}")
lines.append(f"  green={best['green_time']}, yellow={best['yellow_time']}, decay={best['epsilon_decay']}")
lines.append(f"  Avg Wait   : {best['avg_wait_per_step']:.2f}s")
lines.append(f"  Improvement: {improvement(best['avg_wait_per_step']):+.1f}% over baseline")

summary = "\n".join(lines)
print("\n" + summary)

with open("plots/summary.txt", "w") as f:
    f.write(summary)

print("\nSaved plots/summary.txt")
print("All done — check the plots/ folder")
