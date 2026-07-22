"""
plot_switch_timing.py
For PPO
Logs the step index of every switch event per junction across several
episodes, then plots a histogram of switch timing over the 3600-step
episode. Labels the plot/filename with the actual config the loaded
model was trained/evaluated with.

Model path is built from train.py's own MODELS_DIR constant (import
train.py directly) rather than a separately hardcoded folder name --
this has already broken twice from hardcoded paths going stale when
the folder name changed (obs_imbalance_feature, then a flat models/
root), so this is the last place it should be typed by hand. Reward
config and min_green are likewise sourced from trainmod's constants,
not retyped or left to env.py's class defaults.

config_str and the run label are built from env.py's get_hyperparams()
-- single source of truth, can't drift out of sync with what the env
actually ran.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from env import SumoTrafficEnv2J

# Source of truth for both the model location AND the training
# hyperparameters -- see module docstring above for why.
import train as trainmod

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(trainmod.MODELS_DIR, "ppo_sumo_2junction")
NORMALIZER_PATH = os.path.join(trainmod.MODELS_DIR, "vec_normalize_sumo.pkl")
MAX_STEPS       = 3600
N_EPISODES      = 5
TL_IDS          = ["J1", "J2"]
BIN_SIZE        = 100


def make_env(seed):
    def _init():
        return SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            use_gui=False,
            max_steps=MAX_STEPS,
            seed=seed,
            switch_penalty=trainmod.SWITCH_PENALTY,
            wasted_vote_penalty=trainmod.WASTED_VOTE_PENALTY,
            imbalance_bonus_weight=trainmod.IMBALANCE_BONUS_WEIGHT,
            wrong_direction_penalty=trainmod.WRONG_DIRECTION_PENALTY,
            min_green=trainmod.MIN_GREEN,
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
    if not os.path.exists(MODEL_PATH + ".zip"):
        sys.exit(f"FATAL: no PPO model found at {MODEL_PATH}.zip -- train it with train.py first")

    base_env = make_vec_env(make_env(seed=0), n_envs=1)
    base_env = VecNormalize.load(NORMALIZER_PATH, base_env)
    base_env.training = False
    base_env.norm_reward = False
    model = PPO.load(MODEL_PATH, env=base_env)

    # Unwrap to the base env and read its ACTUAL effective config via
    # get_hyperparams() -- guaranteed to match what make_env() above
    # actually passed in, since both come from trainmod.
    sample_env = base_env.envs[0]
    while hasattr(sample_env, "env"):   # unwrap VecEnv/Monitor wrappers if present
        sample_env = sample_env.env
    hp = sample_env.get_hyperparams()
    config_str = (
        f"switch_penalty={hp['switch_penalty']}, "
        f"wrong_direction_penalty={hp['wrong_direction_penalty']}, "
        f"wasted_vote_penalty={hp['wasted_vote_penalty']}, "
        f"imbalance_bonus_weight={hp['imbalance_bonus_weight']}, "
        f"min_green={hp['min_green']}"
    )
    run_label = f"sp{hp['switch_penalty']}_wd{hp['wrong_direction_penalty']}_mingreen{hp['min_green']}"
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

    fig.suptitle(
        f"Switch timing — {run_label}\n{config_str}\ngenerated {timestamp}",
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
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_filename = f"switch_timing_plot_{run_label}_{timestamp}.png"
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