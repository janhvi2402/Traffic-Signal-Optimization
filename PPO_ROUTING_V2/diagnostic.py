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
  - MIN_GREEN is now RANDOMIZED per episode (MIN_GREEN_RANGE=(10, 20),
    set in single_env.py), not a fixed constant. The switch-rate-vs-
    ceiling stat below uses the MEAN of the min_green values actually
    sampled across this run's episodes, not a hardcoded number.
  - NEW, added after diagnosing a policy collapse this script's OLD
    output missed: a "vote rate" section tracking the RAW action
    returned by model.predict() (before the env's eligibility gating),
    split into votes cast while eligible vs while ineligible. Previously
    this script only ever saw the GATED outcome (hold_duration_at_switch,
    which always looks reasonable-ish even when the raw vote is nearly
    constant, since the env still only lets a switch land once eligible
    either way). The vote-rate numbers are what actually distinguish
    "the policy is making decisions" from "the policy always votes 1 and
    lets MIN_GREEN silently gate everything" -- a wasted-vote rate near
    100% (of all steps where a vote was even possible, i.e. steps 1..
    MIN_GREEN-1 of every phase) means the raw vote carries no
    information, REGARDLESS of how reasonable hold_duration looks.
  - "Slack" (mean hold duration minus mean min_green sampled) is now
    printed explicitly instead of requiring you to subtract two numbers
    from the summary by hand -- slack near 0 means switches are firing
    the instant they're legally eligible, not when queue state actually
    warrants it.
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
        "min_green_used": env.MIN_GREEN,   # randomized per episode in reset()
        "votes_cast_while_ineligible": 0,  # raw action==1 while not eligible/not yellow
        "ineligible_steps": 0,             # total steps where a vote COULD be wasted
    }
    current_hold_imbalance = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_int = int(action)

        was_eligible = (not env._in_yellow) and (env._time_in_phase >= env.MIN_GREEN)
        was_in_yellow = env._in_yellow
        if not was_in_yellow and not was_eligible:
            log["ineligible_steps"] += 1
            if action_int == 1:
                log["votes_cast_while_ineligible"] += 1

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
           "wrong_direction_count": 0, "n_switches": 0, "min_greens_used": [],
           "votes_cast_while_ineligible": 0, "ineligible_steps": 0}
    for log in all_logs:
        agg["hold_durations"].extend(log["hold_durations"])
        agg["hold_mean_abs_imbalance"].extend(log["hold_mean_abs_imbalance"])
        agg["agrees"].extend(log["switch_agrees_with_imbalance"])
        agg["wrong_direction_count"] += log["wrong_direction_count"]
        agg["n_switches"] += log["n_switches"]
        agg["min_greens_used"].append(log["min_green_used"])
        agg["votes_cast_while_ineligible"] += log["votes_cast_while_ineligible"]
        agg["ineligible_steps"] += log["ineligible_steps"]
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
    mean_min_green = float(np.mean(agg["min_greens_used"]))
    # ceiling uses the MEAN of the min_green values actually sampled
    # this run (MIN_GREEN is randomized per episode, not fixed) --
    # exact per-episode ceilings vary slightly with each episode's draw
    max_switches_possible = MAX_STEPS / (mean_min_green + SumoSingleJunctionEnv.YELLOW_TIME)

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
    mean_hold = durations.mean() if len(durations) else float("nan")
    slack = mean_hold - mean_min_green
    vote_rate_pct = (
        100 * agg["votes_cast_while_ineligible"] / agg["ineligible_steps"]
        if agg["ineligible_steps"] > 0 else float("nan")
    )

    print("\n" + "=" * 70)
    print(f"SINGLE-JUNCTION DIAGNOSTIC (MIN_GREEN_RANGE={SumoSingleJunctionEnv.MIN_GREEN_RANGE}, "
          f"mean sampled this run={mean_min_green:.1f}, "
          f"switch_penalty={SumoSingleJunctionEnv.SWITCH_PENALTY}, "
          f"wasted_vote_penalty={SumoSingleJunctionEnv.WASTED_VOTE_PENALTY}, "
          f"imbalance_bonus_weight={SumoSingleJunctionEnv.IMBALANCE_BONUS_WEIGHT}, "
          f"wrong_direction_penalty={SumoSingleJunctionEnv.WRONG_DIRECTION_PENALTY})")
    print("=" * 70)
    print(f"Total switches across {N_EPISODES} episodes: {n_switches}")
    print(f"Switch rate vs physical ceiling:       {switch_pct:.1f}%")
    print(f"Mean hold duration:                    {mean_hold:.1f} steps")
    print(f"Slack (mean hold - mean min_green):    {slack:+.1f} steps")
    print("  -> near 0 means switches fire the INSTANT they're legally eligible,")
    print("     regardless of queue state -- this is the single clearest sign of")
    print("     a policy that's still letting MIN_GREEN gate everything for it.")
    print(f"Wasted-vote rate (votes cast while ineligible / all ineligible steps): {vote_rate_pct:.1f}%")
    print("  -> near 100% means the raw vote is ~always 1 regardless of the")
    print("     observation -- a POLICY COLLAPSE, independent of how the gated")
    print("     hold-duration numbers below look. This is the most direct check.")
    print(f"Hold-duration / imbalance correlation: {corr:.3f}")
    print("  -> positive => held longer when its side was genuinely busier")
    print("  -> CAUTION: if slack is near 0, this correlation can be a CONFOUND")
    print("     of MIN_GREEN's own randomization, not evidence of learning --")
    print("     check slack and wasted-vote rate FIRST.")
    print(f"Directional agreement at switch:       {agree_pct:.1f}%")
    print("  -> should be well above 50% if the policy is using queue state")
    print(f"Wrong-direction rate:                  {wrong_pct:.1f}%")
    print("  -> should be low -- this is exactly what wrong_direction_penalty targets")
    print("=" * 70)


if __name__ == "__main__":
    main()