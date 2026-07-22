"""
plot_switch_timing_multi.py

Same diagnostic as plot_switch_timing.py (histogram of switch-event
timing across the episode, per junction) generalized to run PPO and
DQN checkpoints side by side in the same figure -- exactly the same
generalization test_diagnostic_multi.py applied to
test_diagnostic_imbalance.py.

plot_switch_timing.py itself is intentionally left untouched and
PPO-only (per its own docstring) for quick single-model checks. Use
THIS script when you want the PPO-vs-DQN comparison.

Model paths, reward config, and MIN_GREEN come from train.py's and
train_dqn.py's own MODELS_DIR / constants (imported, not retyped) --
same reasoning as compare_ppo_dqn_fixed.py and test_diagnostic_multi.py:
a hardcoded path/config here has already gone stale twice before.

DQN runs go through FlattenMultiDiscreteAction so model.predict() sees
the Discrete(4) action space it was trained on; the underlying env's
info["switched"] dict is unaffected by that wrapper (it only touches
actions), so switch-event logging stays apples-to-apples between the
two algorithms.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from env import SumoTrafficEnv2J
from wrappers import FlattenMultiDiscreteAction

# Source of truth for model locations, reward config, and MIN_GREEN.
import train as trainmod
import train_dqn as traindqnmod

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_STEPS  = 3600
N_EPISODES = 5
TL_IDS     = ["J1", "J2"]
BIN_SIZE   = 100

assert trainmod.MIN_GREEN == traindqnmod.MIN_GREEN, (
    f"train.py MIN_GREEN={trainmod.MIN_GREEN} != "
    f"train_dqn.py MIN_GREEN={traindqnmod.MIN_GREEN} -- fix one before comparing."
)

# (label prefix, source module, algo tag) -- mirrors test_diagnostic_multi.py's
# SWEEP_CONFIGS pattern. Add more entries here (e.g. legacy checkpoints)
# the same way test_diagnostic_multi.py does, if you want them on the
# same plot later.
RUN_CONFIGS = [
    ("PPO", trainmod, "ppo"),
    ("DQN", traindqnmod, "dqn"),
]


def make_env(seed, mod, wrap_discrete):
    def _init():
        env = SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            use_gui=False,
            max_steps=MAX_STEPS,
            seed=seed,
            switch_penalty=mod.SWITCH_PENALTY,
            wasted_vote_penalty=mod.WASTED_VOTE_PENALTY,
            imbalance_bonus_weight=mod.IMBALANCE_BONUS_WEIGHT,
            wrong_direction_penalty=mod.WRONG_DIRECTION_PENALTY,
            min_green=mod.MIN_GREEN,
        )
        if wrap_discrete:
            env = FlattenMultiDiscreteAction(env)
        return env
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


def evaluate_config(label, mod, algo):
    wrap_discrete = (algo == "dqn")
    algo_cls = DQN if algo == "dqn" else PPO
    model_name = "dqn_sumo_2junction" if algo == "dqn" else "ppo_sumo_2junction"
    norm_name = "vec_normalize_sumo_dqn.pkl" if algo == "dqn" else "vec_normalize_sumo.pkl"

    model_path = os.path.join(mod.MODELS_DIR, model_name)
    norm_path  = os.path.join(mod.MODELS_DIR, norm_name)

    if not os.path.exists(model_path + ".zip"):
        print(f"[skip] {label}: no model found at {model_path}.zip")
        return None

    base_env = make_vec_env(make_env(seed=0, mod=mod, wrap_discrete=wrap_discrete), n_envs=1)
    base_env = VecNormalize.load(norm_path, base_env)
    base_env.training = False
    base_env.norm_reward = False
    model = algo_cls.load(model_path, env=base_env)

    # Read the ACTUAL effective config via get_hyperparams(), same
    # guarantee as the other scripts -- can't drift from what the env
    # actually ran.
    sample_env = base_env.envs[0]
    while hasattr(sample_env, "env"):
        sample_env = sample_env.env
    hp = sample_env.get_hyperparams()
    config_str = (
        f"switch_penalty={hp['switch_penalty']}, "
        f"wrong_direction_penalty={hp['wrong_direction_penalty']}, "
        f"wasted_vote_penalty={hp['wasted_vote_penalty']}, "
        f"imbalance_bonus_weight={hp['imbalance_bonus_weight']}, "
        f"min_green={hp['min_green']}"
    )
    base_env.close()

    all_switch_steps = {tl: [] for tl in TL_IDS}
    for ep in range(N_EPISODES):
        env_raw = make_vec_env(make_env(seed=ep, mod=mod, wrap_discrete=wrap_discrete), n_envs=1)
        env = VecNormalize.load(norm_path, env_raw)
        env.training = False
        env.norm_reward = False
        switch_steps = run_episode(model, env)
        for tl in TL_IDS:
            all_switch_steps[tl].extend(switch_steps[tl])
        env.close()
        print(f"  [{label}] episode {ep} done — J1: {len(switch_steps['J1'])} switches, "
              f"J2: {len(switch_steps['J2'])} switches")

    return {"label": label, "config_str": config_str, "switch_steps": all_switch_steps}


def plot_comparison(results, timestamp):
    results = [r for r in results if r is not None]
    if not results:
        print("\nNo results to plot — check [skip] messages above.\n")
        return

    bins = np.arange(0, MAX_STEPS + BIN_SIZE, BIN_SIZE)

    # Grid: rows = junctions, columns = algos (results) -- keeps each
    # algo's histogram uncluttered instead of overlaying, since switch
    # counts can differ a lot in scale between algorithms.
    fig, axes = plt.subplots(len(TL_IDS), len(results), figsize=(6 * len(results), 4.5 * len(TL_IDS)),
                              sharex=True, squeeze=False)

    fig.suptitle(f"Switch timing comparison — generated {timestamp}", fontsize=11)

    for col, r in enumerate(results):
        for row, tl in enumerate(TL_IDS):
            ax = axes[row][col]
            steps = r["switch_steps"][tl]
            counts, _ = np.histogram(steps, bins=bins)
            ax.hist(steps, bins=bins, color="steelblue", edgecolor="white")
            ax.set_title(f"{r['label']} — {tl} (n={len(steps)})", fontsize=10)
            if counts.sum() > 0:
                peak_bin = bins[np.argmax(counts)]
                ax.axvline(peak_bin, color="red", linestyle="--", linewidth=1,
                           label=f"peak: step {peak_bin}-{peak_bin+BIN_SIZE}")
                ax.legend(fontsize=8)
            if col == 0:
                ax.set_ylabel("switch count per 100-step bin")
            if row == len(TL_IDS) - 1:
                ax.set_xlabel("step within episode")

    # Print each config underneath so it's clear what each column is,
    # same guarantee as the single-model script: read off get_hyperparams(),
    # not trusted from a label string.
    config_lines = "\n".join(f"{r['label']}: {r['config_str']}" for r in results)
    fig.text(0.01, 0.01, config_lines, fontsize=7, va="bottom")

    plt.tight_layout(rect=[0, 0.06, 1, 0.93])

    out_filename = f"switch_timing_plot_multi_{timestamp}.png"
    out_path = os.path.join(SCRIPT_DIR, out_filename)
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot -> {out_path}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results = []
    for label, mod, algo in RUN_CONFIGS:
        print(f"\nEvaluating: {label} ({mod.MODELS_DIR}) ...")
        results.append(evaluate_config(label, mod, algo))

    plot_comparison(results, timestamp)

    for r in results:
        if r is None:
            continue
        print(f"\nConfig ({r['label']}): {r['config_str']}")
        bins = np.arange(0, MAX_STEPS + BIN_SIZE, BIN_SIZE)
        for tl in TL_IDS:
            counts, _ = np.histogram(r["switch_steps"][tl], bins=bins)
            cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else float("nan")
            print(f"  {tl}: coefficient of variation across time bins = {cv:.3f}")


if __name__ == "__main__":
    main()