"""
test_diagnostic_imbalance.py

Checks whether the trained policy is actually conditioning on queue
imbalance, or just cycling at the MIN_GREEN ceiling regardless of state.

Three checks:
1. Switch rate vs physical ceiling (quick sanity check, same as before)
2. Correlation: hold duration vs mean |imbalance| during that hold.
   If the agent is reading imbalance, longer holds should coincide
   with the phase's side being clearly busier (i.e. it holds NS green
   longer specifically when NS is heavier than EW).
3. Directional agreement: at each switch, was the NEW phase the side
   that had the larger queue in the steps leading up to it? A policy
   ignoring state should sit near 50% (coin flip). A policy reading
   state should be well above 50%.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from env import SumoTrafficEnv2J

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(SCRIPT_DIR, "models", "ppo_sumo_2junction")
NORMALIZER_PATH = os.path.join(SCRIPT_DIR, "models", "vec_normalize_sumo.pkl")
MAX_STEPS       = 3600
N_EPISODES      = 5

TL_IDS = ["J1", "J2"]


def make_env(seed):
    def _init():
        return SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            use_gui=False,
            max_steps=MAX_STEPS,
            seed=seed,
        )
    return _init


def run_episode(model, env, seed):
    obs = env.reset()
    done = False

    # per-junction logs
    log = {tl: {
        "imbalance": [],          # signed ns_queue - ew_queue, every step
        "phase": [],
        "hold_durations": [],     # duration of each completed hold
        "hold_mean_abs_imbalance": [],  # mean |imbalance| during that hold
        "switch_agrees_with_imbalance": [],  # bool per switch
    } for tl in TL_IDS}

    # running accumulator for "current hold" imbalance
    current_hold_imbalance = {tl: [] for tl in TL_IDS}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        info = info[0]  # vec env wraps info in a list

        for tl in TL_IDS:
            imb = info["imbalance"][tl]
            phase = info["phase"][tl]
            log[tl]["imbalance"].append(imb)
            log[tl]["phase"].append(phase)
            current_hold_imbalance[tl].append(imb)

            hold_dur = info["hold_duration_at_switch"][tl]
            if hold_dur is not None:
                # a switch just fired for this tl — close out the hold
                mean_abs_imb = float(np.mean(np.abs(current_hold_imbalance[tl])))
                log[tl]["hold_durations"].append(hold_dur)
                log[tl]["hold_mean_abs_imbalance"].append(mean_abs_imb)

                # directional check: was the NEW phase the side that was
                # actually busier during the hold that just ended?
                mean_signed_imb = float(np.mean(current_hold_imbalance[tl]))
                old_phase = 1 - phase  # phase already flipped by env at this point... 
                # NOTE: 'phase' in info is the phase AFTER the switch resolves
                # to green; hold_duration_at_switch fires the step the switch
                # is INITIATED (still old phase, entering yellow). We use the
                # sign of imbalance during the hold to check whether the
                # upcoming new phase matches the busier side.
                new_side_is_ns = (old_phase == 1)  # switching FROM ew-green -> ns-green
                agrees = (mean_signed_imb > 0) == new_side_is_ns
                log[tl]["switch_agrees_with_imbalance"].append(agrees)

                current_hold_imbalance[tl] = []

    return log


def aggregate(all_logs):
    agg = {tl: {
        "hold_durations": [],
        "hold_mean_abs_imbalance": [],
        "agrees": [],
    } for tl in TL_IDS}

    for log in all_logs:
        for tl in TL_IDS:
            agg[tl]["hold_durations"].extend(log[tl]["hold_durations"])
            agg[tl]["hold_mean_abs_imbalance"].extend(log[tl]["hold_mean_abs_imbalance"])
            agg[tl]["agrees"].extend(log[tl]["switch_agrees_with_imbalance"])

    return agg


def main():
    base_env = make_vec_env(make_env(seed=0), n_envs=1)
    base_env = VecNormalize.load(NORMALIZER_PATH, base_env)
    base_env.training = False
    base_env.norm_reward = False
    model = PPO.load(MODEL_PATH, env=base_env)

    all_logs = []
    for ep in range(N_EPISODES):
        env_raw = make_vec_env(make_env(seed=ep), n_envs=1)
        env = VecNormalize.load(NORMALIZER_PATH, env_raw)
        env.training = False
        env.norm_reward = False
        log = run_episode(model, env, seed=ep)
        all_logs.append(log)
        env.close()
        print(f"episode {ep} done")

    agg = aggregate(all_logs)

    max_switches_possible = MAX_STEPS / (SumoTrafficEnv2J.MIN_GREEN + SumoTrafficEnv2J.YELLOW_TIME)

    print("\n" + "=" * 70)
    print("SWITCH RATE vs CEILING")
    print("=" * 70)
    for tl in TL_IDS:
        n_switches = len(agg[tl]["hold_durations"])
        pct = 100 * n_switches / (N_EPISODES * max_switches_possible)
        print(f"{tl}: {n_switches} switches over {N_EPISODES} episodes "
              f"({pct:.1f}% of physical max)")

    print("\n" + "=" * 70)
    print("HOLD DURATION vs IMBALANCE CORRELATION")
    print("(if agent reads state: longer holds should correlate with")
    print(" higher |imbalance| during that hold — it's staying green")
    print(" longer because that side genuinely is busier)")
    print("=" * 70)
    for tl in TL_IDS:
        durations = np.array(agg[tl]["hold_durations"])
        imbalances = np.array(agg[tl]["hold_mean_abs_imbalance"])
        if len(durations) > 2 and np.std(durations) > 0 and np.std(imbalances) > 0:
            corr = np.corrcoef(durations, imbalances)[0, 1]
        else:
            corr = float("nan")
        print(f"{tl}: correlation = {corr:.3f}  "
              f"(mean hold={durations.mean():.1f} steps, "
              f"mean |imbalance|={imbalances.mean():.2f} veh)")
        if abs(corr) < 0.15:
            print(f"     -> near zero: hold length looks INDEPENDENT of imbalance")
        elif corr > 0.15:
            print(f"     -> positive: agent DOES hold longer when more imbalanced")

    print("\n" + "=" * 70)
    print("DIRECTIONAL AGREEMENT AT SWITCH TIME")
    print("(does the agent switch TO the side that was actually busier?")
    print(" 50% = coin flip / ignoring state. Meaningfully >50% = reading state)")
    print("=" * 70)
    for tl in TL_IDS:
        agrees = np.array(agg[tl]["agrees"])
        pct = 100 * agrees.mean() if len(agrees) else float("nan")
        print(f"{tl}: {pct:.1f}% of switches went to the side with the "
              f"larger mean queue during the preceding hold "
              f"(n={len(agrees)})")

    # plot: hold duration vs mean |imbalance| scatter, one subplot per junction
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, tl in zip(axes, TL_IDS):
        durations = np.array(agg[tl]["hold_durations"])
        imbalances = np.array(agg[tl]["hold_mean_abs_imbalance"])
        ax.scatter(imbalances, durations, alpha=0.4, s=15)
        ax.set_xlabel("mean |imbalance| during hold (veh)")
        ax.set_ylabel("hold duration (steps)")
        ax.set_title(f"{tl}: hold duration vs imbalance")
        ax.axhline(SumoTrafficEnv2J.MIN_GREEN, color="red", linestyle="--",
                   linewidth=1, label="MIN_GREEN")
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "diagnostic_imbalance_plot.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot -> {out_path}")


if __name__ == "__main__":
    main()