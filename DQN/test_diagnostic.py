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

IMPORTANT CAVEATS, same as before:
- sweep_sp* PPO models were trained WITHOUT the seed-rotation fix.
- The current best PPO model (models/) was trained WITH seed rotation
  AND wrong_direction_penalty in the same run.
- The DQN model, once trained with train_dqn.py, is matched to that
  same reward config and timestep budget -- so DQN vs current-best-PPO
  is the one comparison here that IS single-variable (algorithm only).
  DQN vs the sweep_sp* PPO models is not (those predate the seed fix).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from env import SumoTrafficEnv2J
from wrappers import FlattenMultiDiscreteAction

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MAX_STEPS   = 3600
N_EPISODES  = 5
TL_IDS      = ["J1", "J2"]

# --- configure which models to compare ---
# (label, models_root, folder, algo)
#   models_root: absolute-or-relative path to the folder CONTAINING the
#                model subfolder (e.g. SCRIPT_DIR/"models" for PPO runs,
#                SCRIPT_DIR/"models_dqn" for the DQN run)
#   folder:      subfolder name under models_root ("" for the root itself)
#   algo:        "ppo" or "dqn"
SWEEP_CONFIGS = [
    # Clean single-variable pair: both trained on the CURRENT env.py
    # (16-dim obs, explicit imbalance feature), same reward config,
    # same seed rotation. Algorithm is the only thing that differs.
    ("PPO obs_imbalance_feature (sp=0.3, wd=0.15)",
     os.path.join(SCRIPT_DIR, "models"), "obs_imbalance_feature", "ppo"),
    ("DQN obs_imbalance_feature (sp=0.3, wd=0.15)",
     os.path.join(SCRIPT_DIR, "models_dqn"), "obs_imbalance_feature", "dqn"),

    # Legacy rows, kept for context only -- NOT comparable to the pair
    # above or each other on obs space, reward config, or seed-rotation
    # status. Skipped automatically (with a printed message) if their
    # model files don't exist. Delete these once you've written up the
    # obs_imbalance_feature result and don't need the history anymore.
    ("legacy: PPO wrong_dir run (14-dim obs, sp=0.3, wd=0.15)",
     os.path.join(SCRIPT_DIR, "models"), "", "ppo"),
    ("legacy: PPO sweep sp=0.15 (14-dim obs, no seed rotation)",
     os.path.join(SCRIPT_DIR, "models"), "sweep_sp015", "ppo"),
    ("legacy: PPO sweep sp=0.3 (14-dim obs, no seed rotation)",
     os.path.join(SCRIPT_DIR, "models"), "sweep_sp03", "ppo"),
    ("legacy: PPO sweep sp=0.5 (14-dim obs, no seed rotation)",
     os.path.join(SCRIPT_DIR, "models"), "sweep_sp05", "ppo"),
]


def make_env(seed, wrap_discrete):
    def _init():
        env = SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            use_gui=False,
            max_steps=MAX_STEPS,
            seed=seed,
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
                new_side_is_ns = (old_phase == 1)
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


def evaluate_config(label, models_root, folder, algo):
    model_dir = os.path.join(models_root, folder) if folder else models_root
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
    max_switches_possible = MAX_STEPS / (SumoTrafficEnv2J.MIN_GREEN + SumoTrafficEnv2J.YELLOW_TIME)

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
    print("DQN row both use seed rotation + wrong_direction_penalty=0.15, so THAT")
    print("pair is the clean single-variable (algorithm-only) comparison. Every")
    print("other pairing in this table mixes more than one changed variable.")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("COMPARISON ACROSS CONFIGS")
    print("=" * 100)
    header = f"{'config':<38}{'tl':<4}{'switch %ceil':>14}{'corr':>10}{'mean hold':>12}{'dir. agree %':>14}"
    print(header)
    print("-" * 100)
    for r in results:
        for tl in TL_IDS:
            d = r["per_tl"][tl]
            print(f"{r['label']:<38}{tl:<4}{d['switch_pct_of_ceiling']:>13.1f}%"
                  f"{d['correlation']:>10.3f}{d['mean_hold']:>12.1f}"
                  f"{d['directional_agreement_pct']:>13.1f}%")
    print("=" * 100)
    print("Watch: as switch_penalty rises, switch % should drop. If directional")
    print("agreement DOESN'T rise alongside it, the penalty is suppressing")
    print("switching in general, not making switches smarter. Also watch")
    print("whether each config clears 50% (coin flip) convincingly on both J1")
    print("and J2 -- your current PPO model clears it more on J1 than J2.")


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
    for label, models_root, folder, algo in SWEEP_CONFIGS:
        print(f"\nEvaluating: {label} ({algo}, {folder or 'root'}) ...")
        result = evaluate_config(label, models_root, folder, algo)
        results.append(result)

    print_comparison_table(results)
    plot_comparison(results)


if __name__ == "__main__":
    main()