"""
test.py

Evaluates the single_env.py-trained model, transferred decentrally onto
multi_env.py (obs split 8/16, same shared model predicts on each half
independently) against the fixed-time baseline, then runs the J1/J2
divergence check.

CHANGES vs your previous eval_decentrakized.py:
  - multi_env.py's MIN_GREEN is now 15 (was 10), matching single_env.py
    and centralized -- no code change needed here since it's read off
    the env, just flagging it since it changes how often switches can
    even occur.
  - Divergence check now also reports the correlation between each
    junction's own |imbalance| and its own switch-vote (not just the
    raw J1/J2 agreement rate), since a high same-action rate alone is
    ambiguous -- it's correct when both junctions are genuinely
    similarly loaded, and only a problem if actions ignore local state
    entirely.
"""

import os
import numpy as np
import traci
from stable_baselines3 import PPO

from multi_env import SumoMultiJunctionEnv
from baseline import run_offset_fixed_time

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(SCRIPT_DIR, "models", "ppo_single_junction")
SUMOCFG_PATH = os.path.join(SCRIPT_DIR, "network", "multi_junction.sumocfg")

RECORD = False


def run_decentralized(model, n_episodes=5, collect_divergence=False):
    """
    Applies the SAME single-junction-trained model independently to J1
    and J2. obs is 16-dim: [J1's 8 features, J2's 8 features].
    """
    episode_waits = []
    divergence_log = []  # (raw_imb_j1, a_j1, raw_imb_j2, a_j2) across all steps
    vote_stats = {"J1": {"votes_ineligible": 0, "ineligible_steps": 0},
                  "J2": {"votes_ineligible": 0, "ineligible_steps": 0}}

    for ep in range(n_episodes):
        env = SumoMultiJunctionEnv(
            cfg_path=SUMOCFG_PATH,
            use_gui=RECORD,
            max_steps=3600,
            seed=ep,
            port=8820,
            randomize_routes=True,
        )
        obs, _ = env.reset(seed=ep)
        done = False
        steps = 0
        wait_sum = 0.0

        while not done:
            obs_j1 = obs[0:8]
            obs_j2 = obs[8:16]

            a_j1, _ = model.predict(obs_j1, deterministic=True)
            a_j2, _ = model.predict(obs_j2, deterministic=True)
            a_j1, a_j2 = int(a_j1), int(a_j2)

            if collect_divergence:
                ns1, ew1 = env._get_raw_ns_ew_queue("J1")
                ns2, ew2 = env._get_raw_ns_ew_queue("J2")
                divergence_log.append((ns1 - ew1, a_j1, ns2 - ew2, a_j2))

                for tl, a in [("J1", a_j1), ("J2", a_j2)]:
                    was_eligible = (not env._in_yellow[tl]) and (env._time_in_phase[tl] >= env._min_green[tl])
                    if not env._in_yellow[tl] and not was_eligible:
                        vote_stats[tl]["ineligible_steps"] += 1
                        if a == 1:
                            vote_stats[tl]["votes_ineligible"] += 1

            if steps % 50 == 0:
                print(
                    f"step {steps}: "
                    f"q_j1={obs_j1[0]:.2f}/{obs_j1[1]:.2f} imb_j1={obs_j1[7]:.2f} a_j1={a_j1} "
                    f"minG_j1={env._min_green['J1']} | "
                    f"q_j2={obs_j2[0]:.2f}/{obs_j2[1]:.2f} imb_j2={obs_j2[7]:.2f} a_j2={a_j2} "
                    f"minG_j2={env._min_green['J2']}"
                )

            obs, reward, done, _, info = env.step([a_j1, a_j2])

            for veh in env.conn.vehicle.getIDList():
                wait_sum += env.conn.vehicle.getWaitingTime(veh)
            steps += 1

        episode_waits.append(wait_sum / steps)
        env.close()

    return np.mean(episode_waits), np.std(episode_waits), divergence_log, vote_stats


def run_fixed_time(n_episodes=5):
    waits = []
    for ep in range(n_episodes):
        cmd = ["sumo-gui" if RECORD else "sumo",
               "-c", SUMOCFG_PATH, "--no-warnings", "--seed", str(ep)]
        if RECORD:
            cmd += ["--start", "--quit-on-end"]
        traci.start(cmd)
        traci.simulationStep()
        _, avg_wait, _, _ = run_offset_fixed_time(max_steps=100000)
        traci.close()
        waits.append(avg_wait)
    return np.mean(waits), np.std(waits)


if __name__ == "__main__":
    model = PPO.load(MODEL_PATH)

    N_EP = 1 if RECORD else 5
    print(f"\nEvaluating decentralized single-junction policy over {N_EP} episode(s)...\n")

    decentral_wait, decentral_std, _, _ = run_decentralized(model, n_episodes=N_EP)
    fixed_wait, fixed_std            = run_fixed_time(n_episodes=N_EP)

    improvement = (fixed_wait - decentral_wait) / fixed_wait * 100

    print(f"{'Metric':<35} {'Fixed-time':>14} {'Decentralized PPO':>18}")
    print("-" * 70)
    print(f"{'Mean avg wait/step (s)':<35} {fixed_wait:>13.2f}s {decentral_wait:>17.2f}s")
    print(f"{'Std':<35} {fixed_std:>13.2f}s {decentral_std:>17.2f}s")
    print(f"\nImprovement over fixed-time: {improvement:.1f}%")

    # --- Divergence check ---
    print("\n--- Divergence check ---")
    _, _, divergence_log, vote_stats = run_decentralized(model, n_episodes=1, collect_divergence=True)

    imb_j1 = np.array([r[0] for r in divergence_log])
    a_j1   = np.array([r[1] for r in divergence_log])
    imb_j2 = np.array([r[2] for r in divergence_log])
    a_j2   = np.array([r[3] for r in divergence_log])

    # policy-collapse check FIRST, same as diagnostic.py -- a high
    # same-action rate or agreement-on-disagreement rate is meaningless
    # to interpret if the raw vote is ~always 1 regardless of local
    # queue state to begin with
    for tl in ["J1", "J2"]:
        s = vote_stats[tl]
        rate = 100 * s["votes_ineligible"] / s["ineligible_steps"] if s["ineligible_steps"] > 0 else float("nan")
        print(f"{tl} wasted-vote rate (votes cast while ineligible / all ineligible steps): {rate:.1f}%")
    print("  -> near 100% on either junction means that junction's raw vote is ~always 1")
    print("     regardless of the observation -- check this BEFORE trusting the")
    print("     divergence numbers below.\n")

    same_action_rate = np.mean(a_j1 == a_j2)
    print(f"J1/J2 same-action rate: {same_action_rate:.1%}  (near 100% is only a "
          f"problem together with a low local correlation below -- see next lines)")

    if len(set(a_j1)) > 1:
        corr_j1 = np.corrcoef(np.abs(imb_j1), a_j1)[0, 1]
        print(f"Correlation(|J1 raw imbalance|, J1 switch-vote): {corr_j1:.3f}")
    else:
        print("J1 only ever chose one action -- can't compute correlation.")

    if len(set(a_j2)) > 1:
        corr_j2 = np.corrcoef(np.abs(imb_j2), a_j2)[0, 1]
        print(f"Correlation(|J2 raw imbalance|, J2 switch-vote): {corr_j2:.3f}")
    else:
        print("J2 only ever chose one action -- can't compute correlation.")

    disagree_mask = np.sign(imb_j1) != np.sign(imb_j2)
    if disagree_mask.sum() > 5:
        agreement_when_disagree = np.mean(a_j1[disagree_mask] == a_j2[disagree_mask])
        print(f"\nSteps where J1/J2 imbalance direction disagrees: {disagree_mask.sum()}")
        print(f"J1/J2 action agreement rate on THOSE steps: {agreement_when_disagree:.1%}")
        print("  -> should be noticeably LOWER than the overall same-action rate above.")
        print("     If it's still ~equal, the policy is ignoring local imbalance direction.")