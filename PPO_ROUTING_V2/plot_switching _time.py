"""
plot_switch_timing.py

Produces two histograms in one figure:
  1. Switch timing on the single junction the model was TRAINED on
     (single_env.py) -- sanity check.
  2. Switch timing for J1 vs J2 under the DECENTRALIZED TRANSFER
     (multi_env.py, same model applied independently to each) -- this is
     the one that answers "are J1 and J2 actually behaving differently",
     visually, alongside test.py's numeric divergence check.

Filename/title carry the live reward config (pulled off the loaded env
instance, not hardcoded) plus a timestamp, matching centralized's
plot_switch_timing.py convention -- so a saved plot always documents
exactly what produced it.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO

from single_env import SumoSingleJunctionEnv
from multi_env import SumoMultiJunctionEnv

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(SCRIPT_DIR, "models", "ppo_single_junction")
SUMOCFG_PATH = os.path.join(SCRIPT_DIR, "network", "multi_junction.sumocfg")
MAX_STEPS    = 3600
N_EPISODES   = 5
BIN_SIZE     = 100

RUN_LABEL = "sp0.15_wd0.2_ibw0.4_mingreen10-20"   # <-- update this each time you retrain


def collect_single_switch_steps(model, n_episodes=N_EPISODES):
    all_steps = []
    for ep in range(n_episodes):
        env = SumoSingleJunctionEnv(use_gui=False, seed=ep, port=8818, randomize_routes=True)
        obs, _ = env.reset()
        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            if info["switched"]:
                all_steps.append(step)
            step += 1
        env.close()
    return all_steps


def collect_multi_switch_steps(model, n_episodes=N_EPISODES):
    steps_by_tl = {"J1": [], "J2": []}
    for ep in range(n_episodes):
        env = SumoMultiJunctionEnv(
            cfg_path=SUMOCFG_PATH, use_gui=False, max_steps=MAX_STEPS,
            seed=ep, port=8822, randomize_routes=True,
        )
        obs, _ = env.reset(seed=ep)
        done = False
        step = 0
        while not done:
            obs_j1, obs_j2 = obs[0:8], obs[8:16]
            a_j1, _ = model.predict(obs_j1, deterministic=True)
            a_j2, _ = model.predict(obs_j2, deterministic=True)
            obs, reward, done, _, info = env.step([int(a_j1), int(a_j2)])
            for tl in ["J1", "J2"]:
                if info["switched"][tl]:
                    steps_by_tl[tl].append(step)
            step += 1
        env.close()
    return steps_by_tl


def main():
    model = PPO.load(MODEL_PATH)

    probe_env = SumoSingleJunctionEnv(use_gui=False, seed=0, port=8819, randomize_routes=True)
    config_str = (
        f"switch_penalty={probe_env.switch_penalty}, "
        f"wrong_direction_penalty={probe_env.wrong_direction_penalty}, "
        f"wasted_vote_penalty={probe_env.wasted_vote_penalty}, "
        f"imbalance_bonus_weight={probe_env.imbalance_bonus_weight}, "
        f"MIN_GREEN_RANGE={probe_env.min_green_range}"
    )
    del probe_env  # never reset/stepped, no sumo connection to close

    print("Collecting single-junction switch timing...")
    single_steps = collect_single_switch_steps(model)
    print(f"  {len(single_steps)} switches across {N_EPISODES} episodes")

    print("Collecting decentralized transfer (J1/J2) switch timing...")
    multi_steps = collect_multi_switch_steps(model)
    print(f"  J1: {len(multi_steps['J1'])} switches, J2: {len(multi_steps['J2'])} switches")

    bins = np.arange(0, MAX_STEPS + BIN_SIZE, BIN_SIZE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Switch timing -- {RUN_LABEL}\n{config_str}\ngenerated {timestamp}", fontsize=10)

    ax = axes[0]
    ax.hist(single_steps, bins=bins, color="steelblue", edgecolor="white")
    ax.set_title(f"Single junction (training env) -- n={len(single_steps)} switches over {N_EPISODES} episodes")
    ax.set_ylabel("switches / 100-step bin")

    colors = {"J1": "steelblue", "J2": "darkorange"}
    for ax, tl in zip(axes[1:], ["J1", "J2"]):
        steps = multi_steps[tl]
        ax.hist(steps, bins=bins, color=colors[tl], edgecolor="white")
        ax.set_title(f"{tl} (decentralized transfer) -- n={len(steps)} switches over {N_EPISODES} episodes")
        ax.set_ylabel("switches / 100-step bin")
        if steps:
            counts, _ = np.histogram(steps, bins=bins)
            peak_bin = bins[np.argmax(counts)]
            ax.axvline(peak_bin, color="red", linestyle="--", linewidth=1,
                       label=f"peak bin: step {peak_bin}-{peak_bin+BIN_SIZE}")
            ax.legend()

    axes[-1].set_xlabel("step within episode")
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_filename = f"switch_timing_plot_{RUN_LABEL}_{timestamp}.png"
    out_path = os.path.join(SCRIPT_DIR, out_filename)
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot -> {out_path}")

    print(f"\nConfig: {config_str}")
    for label, steps in [("single", single_steps), ("J1", multi_steps["J1"]), ("J2", multi_steps["J2"])]:
        if steps:
            counts, _ = np.histogram(steps, bins=bins)
            cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else float("nan")
            print(f"{label}: coefficient of variation across time bins = {cv:.3f}")

    # J1 vs J2 timing similarity: if the decentralized transfer is truly
    # reacting to local state (not a shared shortcut), their per-bin
    # switch-count profiles need not match closely.
    counts_j1, _ = np.histogram(multi_steps["J1"], bins=bins)
    counts_j2, _ = np.histogram(multi_steps["J2"], bins=bins)
    if counts_j1.std() > 0 and counts_j2.std() > 0:
        timing_corr = np.corrcoef(counts_j1, counts_j2)[0, 1]
        print(f"\nJ1 vs J2 switch-timing-profile correlation: {timing_corr:.3f}")
        print("  -> very high (near 1.0) suggests both junctions are switching on the")
        print("     same schedule regardless of their own local demand")


if __name__ == "__main__":
    main()