"""
compare_ppo_dqn_fixed.py

Three-way comparison: fixed-time baseline vs PPO vs DQN, using the
IDENTICAL evaluation protocol for all three (same per-episode seeds,
same metric -- mean avg-wait/step -- same N_EP).

Skips a model gracefully (with a printed message) if it hasn't been
trained yet, so you can run this right now and just get the fixed-time
number, then re-run once dqn_sumo_2junction.zip exists.

Model paths are imported directly from train.py / train_dqn.py's own
MODELS_DIR constants rather than retyped here -- this exact class of
bug (a hardcoded path here going stale when the training script's
output folder changed) has already happened twice with different
wrong folder names. Importing means it structurally can't happen again:
whatever folder the training scripts actually save to is where this
script looks, automatically.
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

# Source of truth for model locations AND MIN_GREEN -- see module
# docstring above.
import train as trainmod
import train_dqn as traindqnmod

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PPO_MODEL_PATH = os.path.join(trainmod.MODELS_DIR, "ppo_sumo_2junction")
PPO_NORM_PATH  = os.path.join(trainmod.MODELS_DIR, "vec_normalize_sumo.pkl")

DQN_MODEL_PATH = os.path.join(traindqnmod.MODELS_DIR, "dqn_sumo_2junction")
DQN_NORM_PATH  = os.path.join(traindqnmod.MODELS_DIR, "vec_normalize_sumo_dqn.pkl")

RECORD = False

# Sanity check at import time: if these two training scripts ever
# specify different MIN_GREEN values, the PPO-vs-DQN comparison below
# is no longer single-variable -- fail loudly here instead of silently
# producing a biased comparison.
assert trainmod.MIN_GREEN == traindqnmod.MIN_GREEN, (
    f"train.py MIN_GREEN={trainmod.MIN_GREEN} != "
    f"train_dqn.py MIN_GREEN={traindqnmod.MIN_GREEN} -- fix one before comparing."
)
MIN_GREEN = trainmod.MIN_GREEN


def make_env(seed=0, wrap_discrete=False):
    def _init():
        env = SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            use_gui=RECORD,
            max_steps=3600,
            seed=seed,
            min_green=MIN_GREEN,
        )
        if wrap_discrete:
            env = FlattenMultiDiscreteAction(env)
        return env
    return _init


def _print_effective_config(vec_env, label):
    """Unwraps to the base env and prints get_hyperparams() so you can
    see, at eval time, exactly what config this run is actually using
    rather than trusting a path name or a comment to be accurate."""
    sample_env = vec_env.envs[0]
    while hasattr(sample_env, "env"):
        sample_env = sample_env.env
    print(f"[{label}] eval env config: {sample_env.get_hyperparams()}")


def run_agent(model, norm_path, n_episodes, wrap_discrete, label):
    episode_waits = []
    for ep in range(n_episodes):
        raw = make_vec_env(make_env(seed=ep, wrap_discrete=wrap_discrete), n_envs=1)
        env = VecNormalize.load(norm_path, raw)
        env.training = False
        env.norm_reward = False
        if ep == 0:
            _print_effective_config(env, label)

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
    print(f"PPO model dir: {trainmod.MODELS_DIR}")
    print(f"DQN model dir: {traindqnmod.MODELS_DIR}\n")

    results = {}

    if os.path.exists(PPO_MODEL_PATH + ".zip"):
        base = make_vec_env(make_env(seed=0, wrap_discrete=False), n_envs=1)
        base = VecNormalize.load(PPO_NORM_PATH, base)
        ppo_model = PPO.load(PPO_MODEL_PATH, env=base)
        base.close()
        results["PPO"] = run_agent(ppo_model, PPO_NORM_PATH, N_EP, wrap_discrete=False, label="PPO")
    else:
        print(f"[skip] no PPO model at {PPO_MODEL_PATH}.zip")

    if os.path.exists(DQN_MODEL_PATH + ".zip"):
        base = make_vec_env(make_env(seed=0, wrap_discrete=True), n_envs=1)
        base = VecNormalize.load(DQN_NORM_PATH, base)
        dqn_model = DQN.load(DQN_MODEL_PATH, env=base)
        base.close()
        results["DQN"] = run_agent(dqn_model, DQN_NORM_PATH, N_EP, wrap_discrete=True, label="DQN")
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