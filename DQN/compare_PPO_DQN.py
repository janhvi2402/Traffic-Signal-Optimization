"""
compare_ppo_dqn_fixed.py

Three-way comparison: fixed-time baseline vs PPO vs DQN, using the
IDENTICAL evaluation protocol for all three (same per-episode seeds,
same metric -- mean avg-wait/step -- same N_EP). This extends your
existing PPO-vs-fixed-time eval script rather than replacing it, so
the PPO number here should match what that script already reports.

Skips a model gracefully (with a printed message) if it hasn't been
trained yet, so you can run this right now and just get the fixed-time
number, then re-run once dqn_sumo_2junction.zip exists.
"""

import os
import sys
import numpy as np
import traci
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from baseline import run_offset_fixed_time

from env import SumoTrafficEnv2J
from wrappers import FlattenMultiDiscreteAction

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Both point at the obs_imbalance_feature run (16-dim obs, current
# env.py). This is the pairing that's a clean single-variable
# (algorithm-only) comparison -- same reward, same obs space, same
# seed rotation. Don't point PPO_MODEL_PATH back at the old models/
# root: that model was trained on the OLD 14-dim obs space and its
# vec_normalize_sumo.pkl won't even load against the current env.
PPO_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "obs_imbalance_feature", "ppo_sumo_2junction")
PPO_NORM_PATH  = os.path.join(SCRIPT_DIR, "models", "obs_imbalance_feature", "vec_normalize_sumo.pkl")

DQN_MODEL_PATH = os.path.join(SCRIPT_DIR, "models_dqn", "obs_imbalance_feature", "dqn_sumo_2junction")
DQN_NORM_PATH  = os.path.join(SCRIPT_DIR, "models_dqn", "obs_imbalance_feature", "vec_normalize_sumo_dqn.pkl")

RECORD = False


def make_env(seed=0, wrap_discrete=False):
    def _init():
        env = SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            use_gui=RECORD,
            max_steps=3600,
            seed=seed,
        )
        if wrap_discrete:
            env = FlattenMultiDiscreteAction(env)
        return env
    return _init


def run_agent(model, norm_path, n_episodes, wrap_discrete):
    episode_waits = []
    for ep in range(n_episodes):
        raw = make_vec_env(make_env(seed=ep, wrap_discrete=wrap_discrete), n_envs=1)
        env = VecNormalize.load(norm_path, raw)
        env.training = False
        env.norm_reward = False

        obs = env.reset()
        done = False
        steps = 0
        wait_sum = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            conn = env.get_attr("conn")[0]
            for veh in conn.vehicle.getIDList():
                wait_sum += conn.vehicle.getWaitingTime(veh)
            steps += 1

        episode_waits.append(wait_sum / steps)
        env.close()

    return np.mean(episode_waits), np.std(episode_waits)


def run_fixed_time(n_episodes):
    waits = []
    for ep in range(n_episodes):
        cmd = ["sumo-gui" if RECORD else "sumo",
               "-c", os.path.join(SCRIPT_DIR, "network.sumocfg"),
               "--no-warnings", "--seed", str(ep)]
        if RECORD:
            cmd += ["--start", "--quit-on-end"]
        traci.start(cmd)
        traci.simulationStep()
        _, avg_wait, _, _ = run_offset_fixed_time(max_steps=100000)
        traci.close()
        waits.append(avg_wait)
    return np.mean(waits), np.std(waits)


def main():
    N_EP = 1 if RECORD else 5
    print(f"\nEvaluating over {N_EP} episode(s) each...\n")

    results = {}

    if os.path.exists(PPO_MODEL_PATH + ".zip"):
        base = make_vec_env(make_env(seed=0, wrap_discrete=False), n_envs=1)
        base = VecNormalize.load(PPO_NORM_PATH, base)
        ppo_model = PPO.load(PPO_MODEL_PATH, env=base)
        base.close()
        results["PPO"] = run_agent(ppo_model, PPO_NORM_PATH, N_EP, wrap_discrete=False)
    else:
        print(f"[skip] no PPO model at {PPO_MODEL_PATH}.zip")

    if os.path.exists(DQN_MODEL_PATH + ".zip"):
        base = make_vec_env(make_env(seed=0, wrap_discrete=True), n_envs=1)
        base = VecNormalize.load(DQN_NORM_PATH, base)
        dqn_model = DQN.load(DQN_MODEL_PATH, env=base)
        base.close()
        results["DQN"] = run_agent(dqn_model, DQN_NORM_PATH, N_EP, wrap_discrete=True)
    else:
        print(f"[skip] no DQN model at {DQN_MODEL_PATH}.zip -- train it with train_dqn.py first")

    fixed_wait, fixed_wait_std = run_fixed_time(N_EP)
    results["Fixed-time"] = (fixed_wait, fixed_wait_std)

    print(f"\n{'Metric':<30}" + "".join(f"{name:>16}" for name in results))
    print("-" * (30 + 16 * len(results)))
    print(f"{'Mean avg wait/step (s)':<30}" +
          "".join(f"{v[0]:>15.2f}s" for v in results.values()))
    print(f"{'Std':<30}" +
          "".join(f"{v[1]:>15.2f}s" for v in results.values()))

    for name in ("PPO", "DQN"):
        if name in results:
            imp = (fixed_wait - results[name][0]) / fixed_wait * 100
            print(f"\nImprovement of {name} over fixed-time: {imp:.1f}%")

    if "PPO" in results and "DQN" in results:
        diff_pct = abs(results["PPO"][0] - results["DQN"][0]) / results["PPO"][0] * 100
        better = "DQN" if results["DQN"][0] < results["PPO"][0] else "PPO"
        print(f"\n{better} has the lower mean wait/step "
              f"(PPO {results['PPO'][0]:.2f}s vs DQN {results['DQN'][0]:.2f}s, "
              f"{diff_pct:.1f}% difference)")


if __name__ == "__main__":
    main()