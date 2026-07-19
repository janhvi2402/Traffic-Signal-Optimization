"""
diagnostic.py

Diagnostic eval for the single_env.py-trained PPO policy, on the same
single junction it was trained on (sanity check before transferring to
multi_env.py -- see test.py for the transfer eval).

CHANGES vs your previous diagnostic script:
  - The old SWITCH_BONUS_WEIGHT mechanism set env.last_switch_was_correct
    at the moment a switch landed; that field no longer exists since the
    reward no longer needs a one-shot "was it correct" event -- WRONG_
    DIRECTION_PENALTY is computed and applied inline instead. This script
    now reads switched/wrong_direction/hold_duration_at_switch straight
    off the info dict returned by step(), matching centralized's
    diagnostic style (hold_durations, hold_mean_abs_imbalance,
    switch_agrees_with_imbalance, wrong_direction_count).
  - MIN_GREEN is now 15 (env.py class constant), used for the switch-
    rate-vs-ceiling stat instead of being hardcoded here.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from single_env import SumoSingleJunctionEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "ppo_single_junction")
MAX_STEPS  = 3600
N_EPISODES = 5


def run_episode(model, env):
    obs, _ = env.reset()
    done = False

    log = {
        "hold_durations": [],
        "hold_mean_abs_imbalance": [],
        "switch_agrees_with_imbalance": [],
        "wrong_direction_count": 0,
        "n_switches": 0,
    }
    current_hold_imbalance = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        imb = info["imbalance"]
        phase = info["phase"]
        current_hold_imbalance.append(imb)

        if info["wrong_direction"]:
            log["wrong_direction_count"] += 1

        hold_dur = info["hold_duration_at_switch"]
        if hold_dur is not None:
            log["n_switches"] += 1
            mean_abs_imb = float(np.mean(np.abs(current_hold_imbalance)))
            mean_signed_imb = float(np.mean(current_hold_imbalance))
            log["hold_durations"].append(hold_dur)
            log["hold_mean_abs_imbalance"].append(mean_abs_imb)

            old_phase = 1 - phase
            new_side_is_ns = (old_phase == 0)
            agrees = (mean_signed_imb > 0) == new_side_is_ns
            log["switch_agrees_with_imbalance"].append(agrees)

            current_hold_imbalance = []

    return log


def aggregate(all_logs):
    agg = {"hold_durations": [], "hold_mean_abs_imbalance": [], "agrees": [],
           "wrong_direction_count": 0, "n_switches": 0}
    for log in all_logs:
        agg["hold_durations"].extend(log["hold_durations"])
        agg["hold_mean_abs_imbalance"].extend(log["hold_mean_abs_imbalance"])
        agg["agrees"].extend(log["switch_agrees_with_imbalance"])
        agg["wrong_direction_count"] += log["wrong_direction_count"]
        agg["n_switches"] += log["n_switches"]
    return agg


def main():
    model = PPO.load(MODEL_PATH)

    all_logs = []
    for ep in range(N_EPISODES):
        env = SumoSingleJunctionEnv(use_gui=False, seed=ep, port=8817, randomize_routes=True)
        log = run_episode(model, env)
        all_logs.append(log)
        env.close()
        print(f"episode {ep} done -- {log['n_switches']} switches, "
              f"{log['wrong_direction_count']} wrong-direction")

    agg = aggregate(all_logs)
    max_switches_possible = MAX_STEPS / (SumoSingleJunctionEnv.MIN_GREEN + SumoSingleJunctionEnv.YELLOW_TIME)

    durations  = np.array(agg["hold_durations"])
    imbalances = np.array(agg["hold_mean_abs_imbalance"])
    agrees     = np.array(agg["agrees"])
    n_switches = agg["n_switches"]

    switch_pct = 100 * n_switches / (N_EPISODES * max_switches_possible)
    if len(durations) > 2 and np.std(durations) > 0 and np.std(imbalances) > 0:
        corr = np.corrcoef(durations, imbalances)[0, 1]
    else:
        corr = float("nan")
    agree_pct = 100 * agrees.mean() if len(agrees) else float("nan")
    wrong_pct = 100 * agg["wrong_direction_count"] / max(n_switches, 1)

    print("\n" + "=" * 70)
    print(f"SINGLE-JUNCTION DIAGNOSTIC (MIN_GREEN={SumoSingleJunctionEnv.MIN_GREEN}, "
          f"switch_penalty={SumoSingleJunctionEnv.SWITCH_PENALTY}, "
          f"wrong_direction_penalty={SumoSingleJunctionEnv.WRONG_DIRECTION_PENALTY})")
    print("=" * 70)
    print(f"Total switches across {N_EPISODES} episodes: {n_switches}")
    print(f"Switch rate vs physical ceiling:       {switch_pct:.1f}%")
    print(f"Mean hold duration:                    {durations.mean() if len(durations) else float('nan'):.1f} steps")
    print(f"Hold-duration / imbalance correlation: {corr:.3f}")
    print("  -> positive => held longer when its side was genuinely busier")
    print(f"Directional agreement at switch:       {agree_pct:.1f}%")
    print("  -> should be well above 50% if the policy is using queue state")
    print(f"Wrong-direction rate:                  {wrong_pct:.1f}%")
    print("  -> should be low -- this is exactly what wrong_direction_penalty targets")
    print("=" * 70)


if __name__ == "__main__":
    main()