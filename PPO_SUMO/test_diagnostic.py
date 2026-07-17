"""
test_diagnostic_imbalance.py

Runs the imbalance diagnostics (switch rate vs ceiling, hold-duration/
imbalance correlation, directional agreement at switch time) across
MULTIPLE trained models in one pass and prints a single comparison
table + plot.

IMPORTANT CAVEAT (read before interpreting results):
The sweep_sp* models were trained WITHOUT the seed-rotation fix — each
was trained on a single repeated SUMO scenario for its entire run. The
"wrong_dir run" model (models/ root) was trained WITH seed rotation
fixed AND with wrong_direction_penalty added, both changed in the same
run. So this table is informative but NOT a clean single-variable
comparison — an improvement in the new model could come from either
change. To isolate wrong_direction_penalty specifically, run one more
config: seed rotation fixed, wrong_direction_penalty=0.0, everything
else matching the new run.

Configure SWEEP_CONFIGS below to point at whatever folders you have.
Each entry needs: a label (for the table/plot) and a folder under
models/ containing ppo_sumo_2junction.zip + vec_normalize_sumo.pkl.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from env import SumoTrafficEnv2J

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(SCRIPT_DIR, "models")
MAX_STEPS   = 3600
N_EPISODES  = 5
TL_IDS      = ["J1", "J2"]

# --- configure which models to compare ---
# label -> folder name under models/. Use "" for the base models/ folder itself.
SWEEP_CONFIGS = [
    ("wrong_dir run (sp=0.3, wd=0.15)", ""),            # UPDATED label — models/ now
                                                           # holds the seed-rotation-fixed,
                                                           # wrong_direction_penalty run,
                                                           # NOT the old sp=0.15 baseline
    ("sweep sp=0.15",                   "sweep_sp015"),  # trained WITHOUT seed rotation fix
    ("sweep sp=0.3",                    "sweep_sp03"),   # trained WITHOUT seed rotation fix
    ("sweep sp=0.5",                    "sweep_sp05"),   # trained WITHOUT seed rotation fix
]


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


def evaluate_config(label, folder):
    model_dir = os.path.join(MODELS_ROOT, folder) if folder else MODELS_ROOT
    model_path = os.path.join(model_dir, "ppo_sumo_2junction")
    norm_path  = os.path.join(model_dir, "vec_normalize_sumo.pkl")

    if not os.path.exists(model_path + ".zip"):
        print(f"[skip] {label}: no model found at {model_path}.zip")
        return None

    base_env = make_vec_env(make_env(seed=0), n_envs=1)
    base_env = VecNormalize.load(norm_path, base_env)
    base_env.training = False
    base_env.norm_reward = False
    model = PPO.load(model_path, env=base_env)
    base_env.close()

    all_logs = []
    for ep in range(N_EPISODES):
        env_raw = make_vec_env(make_env(seed=ep), n_envs=1)
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
    print("NOTE: sweep_sp* models were trained WITHOUT the seed-rotation fix")
    print("(single repeated scenario per run). The first row was trained WITH")
    print("seed rotation fixed AND wrong_direction_penalty added — both changed")
    print("in the same run, so this is NOT a clean single-variable comparison.")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("COMPARISON ACROSS CONFIGS")
    print("=" * 100)
    header = f"{'config':<32}{'tl':<4}{'switch %ceil':>14}{'corr':>10}{'mean hold':>12}{'dir. agree %':>14}"
    print(header)
    print("-" * 100)
    for r in results:
        for tl in TL_IDS:
            d = r["per_tl"][tl]
            print(f"{r['label']:<32}{tl:<4}{d['switch_pct_of_ceiling']:>13.1f}%"
                  f"{d['correlation']:>10.3f}{d['mean_hold']:>12.1f}"
                  f"{d['directional_agreement_pct']:>13.1f}%")
    print("=" * 100)
    print("Watch: as switch_penalty rises, switch % should drop. If")
    print("directional agreement DOESN'T rise alongside it, the penalty")
    print("is suppressing switching in general, not making switches smarter.")
    print("Also watch whether the new run (seed rotation + wrong_direction_penalty)")
    print("clears 50% (coin flip) more convincingly than the old sweep configs did.")


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
    out_path = os.path.join(SCRIPT_DIR, "diagnostic_sweep_comparison.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved comparison plot -> {out_path}")


def main():
    results = []
    for label, folder in SWEEP_CONFIGS:
        print(f"\nEvaluating: {label} ({folder or 'models/'}) ...")
        result = evaluate_config(label, folder)
        results.append(result)

    print_comparison_table(results)
    plot_comparison(results)


if __name__ == "__main__":
    main()