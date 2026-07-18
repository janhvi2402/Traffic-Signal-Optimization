"""
plot_switch_timing.py

Logs the step index of every switch event per junction across several
episodes, then plots a histogram of switch timing over the 3600-step
episode. Now labels the plot/filename with the actual reward config
pulled live from env.py's constants, so you can't lose track of which
run a given plot belongs to.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from env import SumoTrafficEnv2J

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(SCRIPT_DIR, "models", "ppo_sumo_2junction")
NORMALIZER_PATH = os.path.join(SCRIPT_DIR, "models", "vec_normalize_sumo.pkl")
MAX_STEPS       = 3600
N_EPISODES      = 5
TL_IDS          = ["J1", "J2"]
BIN_SIZE        = 100

# NEW: label this run manually if you want a custom note (e.g. which
# machine, or a short description) — combined automatically with the
# live reward-config values read from env.py below.
RUN_LABEL = "sp0.4_wd0.25_mingreen15"   # <-- update this each time you retrain


def make_env(seed):
    def _init():
        return SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            use_gui=False,
            max_steps=MAX_STEPS,
            seed=seed,
        )
    return _init


def run_episode(model, env):
    obs = env.reset()
    done = False
    step = 0
    switch_steps = {tl: [] for tl in TL_IDS}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        info = info[0]

        for tl in TL_IDS:
            if info["switched"][tl]:
                switch_steps[tl].append(step)
        step += 1

    return switch_steps


def main():
    base_env = make_vec_env(make_env(seed=0), n_envs=1)
    base_env = VecNormalize.load(NORMALIZER_PATH, base_env)
    base_env.training = False
    base_env.norm_reward = False
    model = PPO.load(MODEL_PATH, env=base_env)

    # NEW: pull the actual reward config off the loaded env instance,
    # not off env.py's class defaults — this is what the model was
    # ACTUALLY trained/evaluated with, since these can be overridden
    # per-instance via constructor args in train.py.
    sample_env = base_env.envs[0]
    while hasattr(sample_env, "env"):   # unwrap VecEnv/Monitor wrappers if present
        sample_env = sample_env.env
    config_str = (
        f"switch_penalty={sample_env.switch_penalty}, "
        f"wrong_direction_penalty={sample_env.wrong_direction_penalty}, "
        f"wasted_vote_penalty={sample_env.wasted_vote_penalty}, "
        f"imbalance_bonus_weight={sample_env.imbalance_bonus_weight}, "
        f"MIN_GREEN={sample_env.MIN_GREEN}"
    )
    base_env.close()

    all_switch_steps = {tl: [] for tl in TL_IDS}

    for ep in range(N_EPISODES):
        env_raw = make_vec_env(make_env(seed=ep), n_envs=1)
        env = VecNormalize.load(NORMALIZER_PATH, env_raw)
        env.training = False
        env.norm_reward = False
        switch_steps = run_episode(model, env)
        for tl in TL_IDS:
            all_switch_steps[tl].extend(switch_steps[tl])
        env.close()
        print(f"episode {ep} done — J1: {len(switch_steps['J1'])} switches, "
              f"J2: {len(switch_steps['J2'])} switches")

    bins = np.arange(0, MAX_STEPS + BIN_SIZE, BIN_SIZE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # NEW: overall figure title carries the run label + config + timestamp
    fig.suptitle(
        f"Switch timing — {RUN_LABEL}\n{config_str}\ngenerated {timestamp}",
        fontsize=10,
    )

    for ax, tl in zip(axes, TL_IDS):
        ax.hist(all_switch_steps[tl], bins=bins, color="steelblue", edgecolor="white")
        ax.set_title(f"{tl}: switch timing across episode (n={len(all_switch_steps[tl])} "
                      f"switches over {N_EPISODES} episodes)")
        ax.set_ylabel("switch count per 100-step bin")
        counts, _ = np.histogram(all_switch_steps[tl], bins=bins)
        peak_bin = bins[np.argmax(counts)]
        ax.axvline(peak_bin, color="red", linestyle="--", linewidth=1,
                   label=f"peak bin: step {peak_bin}-{peak_bin+BIN_SIZE}")
        ax.legend()

    axes[-1].set_xlabel("step within episode")
    plt.tight_layout(rect=[0, 0, 1, 0.92])   # leave room for suptitle

    # NEW: filename now includes the run label + timestamp so reruns
    # never silently overwrite a previous plot
    out_filename = f"switch_timing_plot_{RUN_LABEL}_{timestamp}.png"
    out_path = os.path.join(SCRIPT_DIR, out_filename)
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot -> {out_path}")

    print(f"\nConfig: {config_str}")
    for tl in TL_IDS:
        counts, _ = np.histogram(all_switch_steps[tl], bins=bins)
        cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else float("nan")
        print(f"{tl}: coefficient of variation across time bins = {cv:.3f}")


if __name__ == "__main__":
    main()