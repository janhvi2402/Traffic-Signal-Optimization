"""
test_diagnostic_multi.py

Same diagnostics as test_diagnostic_imbalance.py (switch rate vs
ceiling, hold-duration/imbalance correlation, directional agreement at
switch time), generalized to run across PPO and DQN checkpoints in the
same table/plot. Each entry in SWEEP_CONFIGS now also names its algo;
DQN entries get routed through FlattenMultiDiscreteAction automatically
so model.predict() sees the Discrete(4) action space it was trained on,
while everything downstream (info dict, imbalance/hold-duration logic)
reads off the *base* env exactly like it did for PPO -- the wrapper
only touches the action, so this comparison stays apples-to-apples.

The main PPO/DQN pair's model directories are imported directly from
train.py / train_dqn.py's own MODELS_DIR constants instead of retyped
as a folder-name string here -- this exact bug (a hardcoded path here
going stale when the training script's output folder changed) has
already happened twice with two different wrong folder names. Only the
LEGACY sweep entries below still use a bare hardcoded folder, since
those are genuinely separate historical runs not tied to what train.py
currently points at.

IMPORTANT CAVEATS:
- sweep_sp* PPO models were trained WITHOUT the seed-rotation fix.
- The current best PPO model and the DQN model are both trained on
  sp=0.3, wd=0.2, MIN_GREEN=12 -- this is the config chosen after
  comparing three empirical runs (see env.py's class docstring for the
  full comparison): it's the only one of the three that showed BOTH a
  real improvement over fixed-time AND above-chance directional
  agreement on both junctions. This PPO/DQN pair is the one clean
  single-variable (algorithm-only) comparison in this table -- every
  other pairing mixes more than one changed variable.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from env import SumoTrafficEnv2J

from wrappers import FlattenMultiDiscreteAction

# Source of truth for the main pair's model locations and MIN_GREEN.
import train as trainmod
import train_dqn as traindqnmod

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MAX_STEPS   = 3600
N_EPISODES  = 5
TL_IDS      = ["J1", "J2"]

assert trainmod.MIN_GREEN == traindqnmod.MIN_GREEN, (
    f"train.py MIN_GREEN={trainmod.MIN_GREEN} != "
    f"train_dqn.py MIN_GREEN={traindqnmod.MIN_GREEN} -- fix one before comparing."
)
MIN_GREEN = trainmod.MIN_GREEN

# --- configure which models to compare ---
# (label, model_dir, algo) -- model_dir is the folder CONTAINING the
# model files directly (not a root+subfolder split), so there's nothing
# left to recombine incorrectly.
SWEEP_CONFIGS = [
    # Clean single-variable pair: same reward config, same MIN_GREEN,
    # same seed rotation -- algorithm is the only thing that differs.
    # Paths come straight from the training scripts' own MODELS_DIR, see
    # module docstring above.
    (f"PPO (sp={trainmod.SWITCH_PENALTY}, wd={trainmod.WRONG_DIRECTION_PENALTY}, MIN_GREEN={trainmod.MIN_GREEN})",
     trainmod.MODELS_DIR, "ppo"),
    (f"DQN (sp={traindqnmod.SWITCH_PENALTY}, wd={traindqnmod.WRONG_DIRECTION_PENALTY}, MIN_GREEN={traindqnmod.MIN_GREEN})",
     traindqnmod.MODELS_DIR, "dqn"),

    # Legacy rows, kept for context only -- NOT comparable to the pair
    # above or each other on reward config or seed-rotation status.
    # Skipped automatically (with a printed message) if their model
    # files don't exist.
    #
    # NOTE: folder names below are PLACEHOLDERS for the two configs from
    # your experiment table -- fill in the real subfolder name under
    # models/ for each before running, or delete the row if you didn't
    # keep that checkpoint.
    ("legacy: PPO sp=0.4, wd=0.25, MIN_GREEN=15 (-17.7%, over-suppressed)",
     os.path.join(SCRIPT_DIR, "models", "<FILL_IN_YOUR_FOLDER_NAME>"), "ppo"),
    ("legacy: PPO sp=0.3, wd=0.15, MIN_GREEN=10 (73.6%, below-chance direction)",
     os.path.join(SCRIPT_DIR, "models", "<FILL_IN_YOUR_FOLDER_NAME>"), "ppo"),
    ("legacy: PPO sweep sp=0.15 (no seed rotation)",
     os.path.join(SCRIPT_DIR, "models", "sweep_sp015"), "ppo"),
    ("legacy: PPO sweep sp=0.3 (no seed rotation)",
     os.path.join(SCRIPT_DIR, "models", "sweep_sp03"), "ppo"),
    ("legacy: PPO sweep sp=0.5 (no seed rotation)",
     os.path.join(SCRIPT_DIR, "models", "sweep_sp05"), "ppo"),
]


def make_env(seed, wrap_discrete):
    def _init():
        env = SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            use_gui=False,
            max_steps=MAX_STEPS,
            seed=seed,
            min_green=MIN_GREEN,
        )
        if wrap_discrete:
            env = FlattenMultiDiscreteAction(env)
        return env
    return _init


def run_episode(model, env):
    obs = env.reset()
    done = False

    log = {tl: {
        "hold_durations": [],
        "hold_mean_abs_imbalance": [],
        "switch_agrees_with_imbalance": [],
    } for tl in TL_IDS}

    current_hold_imbalance = {tl: [] for tl in TL_IDS}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        info = info[0]

        for tl in TL_IDS:
            imb = info["imbalance"][tl]
            phase = info["phase"][tl]
            current_hold_imbalance[tl].append(imb)

            hold_dur = info["hold_duration_at_switch"][tl]
            if hold_dur is not None:
                mean_abs_imb = float(np.mean(np.abs(current_hold_imbalance[tl])))
                mean_signed_imb = float(np.mean(current_hold_imbalance[tl]))
                log[tl]["hold_durations"].append(hold_dur)
                log[tl]["hold_mean_abs_imbalance"].append(mean_abs_imb)

                old_phase = 1 - phase
                new_side_is_ns = (old_phase == 0)
                agrees = (mean_signed_imb > 0) == new_side_is_ns
                log[tl]["switch_agrees_with_imbalance"].append(agrees)

                current_hold_imbalance[tl] = []

    return log


def aggregate(all_logs):
    agg = {tl: {"hold_durations": [], "hold_mean_abs_imbalance": [], "agrees": []}
           for tl in TL_IDS}
    for log in all_logs:
        for tl in TL_IDS:
            agg[tl]["hold_durations"].extend(log[tl]["hold_durations"])
            agg[tl]["hold_mean_abs_imbalance"].extend(log[tl]["hold_mean_abs_imbalance"])
            agg[tl]["agrees"].extend(log[tl]["switch_agrees_with_imbalance"])
    return agg


def evaluate_config(label, model_dir, algo):
    norm_name = "vec_normalize_sumo_dqn.pkl" if algo == "dqn" else "vec_normalize_sumo.pkl"
    model_name = "dqn_sumo_2junction" if algo == "dqn" else "ppo_sumo_2junction"
    model_path = os.path.join(model_dir, model_name)
    norm_path  = os.path.join(model_dir, norm_name)
    wrap_discrete = (algo == "dqn")
    algo_cls = DQN if algo == "dqn" else PPO

    if not os.path.exists(model_path + ".zip"):
        print(f"[skip] {label}: no model found at {model_path}.zip")
        return None

    base_env = make_vec_env(make_env(seed=0, wrap_discrete=wrap_discrete), n_envs=1)
    base_env = VecNormalize.load(norm_path, base_env)
    base_env.training = False
    base_env.norm_reward = False
    model = algo_cls.load(model_path, env=base_env)

    # Verify the effective config this eval env is actually running
    # with, via get_hyperparams(), instead of trusting the folder name
    # / label string to be accurate.
    sample_env = base_env.envs[0]
    while hasattr(sample_env, "env"):
        sample_env = sample_env.env
    print(f"  effective config: {sample_env.get_hyperparams()}")
    base_env.close()

    all_logs = []
    for ep in range(N_EPISODES):
        env_raw = make_vec_env(make_env(seed=ep, wrap_discrete=wrap_discrete), n_envs=1)
        env = VecNormalize.load(norm_path, env_raw)
        env.training = False
        env.norm_reward = False
        log = run_episode(model, env)
        all_logs.append(log)
        env.close()

    agg = aggregate(all_logs)
    max_switches_possible = MAX_STEPS / (MIN_GREEN + SumoTrafficEnv2J.YELLOW_TIME)

    result = {"label": label, "per_tl": {}}
    for tl in TL_IDS:
        durations = np.array(agg[tl]["hold_durations"])
        imbalances = np.array(agg[tl]["hold_mean_abs_imbalance"])
        agrees = np.array(agg[tl]["agrees"])

        n_switches = len(durations)
        switch_pct = 100 * n_switches / (N_EPISODES * max_switches_possible)

        if len(durations) > 2 and np.std(durations) > 0 and np.std(imbalances) > 0:
            corr = np.corrcoef(durations, imbalances)[0, 1]
        else:
            corr = float("nan")

        agree_pct = 100 * agrees.mean() if len(agrees) else float("nan")

        result["per_tl"][tl] = {
            "n_switches": n_switches,
            "switch_pct_of_ceiling": switch_pct,
            "correlation": corr,
            "mean_hold": durations.mean() if len(durations) else float("nan"),
            "directional_agreement_pct": agree_pct,
            "hold_durations": durations,
            "hold_mean_abs_imbalance": imbalances,
        }

    return result


def print_comparison_table(results):
    results = [r for r in results if r is not None]

    print("\n" + "=" * 100)
    print("NOTE: sweep_sp* PPO models were trained WITHOUT the seed-rotation fix")
    print("(single repeated scenario per run). The current-best PPO row and the")
    print(f"DQN row both use seed rotation + sp={trainmod.SWITCH_PENALTY}, "
          f"wd={trainmod.WRONG_DIRECTION_PENALTY}, MIN_GREEN={trainmod.MIN_GREEN}, so THAT")
    print("pair is the clean single-variable (algorithm-only) comparison. Every")
    print("other pairing in this table mixes more than one changed variable.")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("COMPARISON ACROSS CONFIGS")
    print("=" * 100)
    header = f"{'config':<60}{'tl':<4}{'switch %ceil':>14}{'corr':>10}{'mean hold':>12}{'dir. agree %':>14}"
    print(header)
    print("-" * 110)
    for r in results:
        for tl in TL_IDS:
            d = r["per_tl"][tl]
            print(f"{r['label']:<60}{tl:<4}{d['switch_pct_of_ceiling']:>13.1f}%"
                  f"{d['correlation']:>10.3f}{d['mean_hold']:>12.1f}"
                  f"{d['directional_agreement_pct']:>13.1f}%")
    print("=" * 110)
    print("Watch: as switch_penalty rises, switch % should drop. If directional")
    print("agreement DOESN'T rise alongside it, the penalty is suppressing")
    print("switching in general, not making switches smarter. Also watch")
    print("whether each config clears 50% (coin flip) convincingly on both J1")
    print("and J2.")


def plot_comparison(results):
    results = [r for r in results if r is not None]
    labels = [r["label"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for tl, marker in zip(TL_IDS, ["o", "s"]):
        switch_pcts = [r["per_tl"][tl]["switch_pct_of_ceiling"] for r in results]
        corrs       = [r["per_tl"][tl]["correlation"] for r in results]
        agree_pcts  = [r["per_tl"][tl]["directional_agreement_pct"] for r in results]

        x = np.arange(len(labels))
        axes[0].plot(x, switch_pcts, marker=marker, label=tl)
        axes[1].plot(x, corrs, marker=marker, label=tl)
        axes[2].plot(x, agree_pcts, marker=marker, label=tl)

    for ax, title, ylabel in zip(
        axes,
        ["Switch rate vs ceiling", "Hold duration / imbalance correlation", "Directional agreement at switch"],
        ["% of physical max", "correlation (r)", "% agreeing with imbalance"],
    ):
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[2].axhline(50, color="red", linestyle="--", linewidth=1, label="coin flip")

    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "diagnostic_sweep_comparison_multi.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved comparison plot -> {out_path}")


def main():
    results = []
    for label, model_dir, algo in SWEEP_CONFIGS:
        print(f"\nEvaluating: {label} ({model_dir}) ...")
        result = evaluate_config(label, model_dir, algo)
        results.append(result)

    print_comparison_table(results)
    plot_comparison(results)


if __name__ == "__main__":
    main()