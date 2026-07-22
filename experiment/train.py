"""
train.py

Reward config for this run (all values explicit below and passed
straight into the env constructor -- do not rely on env.py's class
defaults, even though they currently match, since defaults can drift):
    SWITCH_PENALTY          = 0.3
    WASTED_VOTE_PENALTY     = 0.03
    IMBALANCE_BONUS_WEIGHT  = 0.0
    WRONG_DIRECTION_PENALTY = 0.2
    MIN_GREEN               = 12

This is the config that produced the 57.5% improvement PPO result
(mean hold ~16-19, J1 direction-agreement ~59.1%, J2 ~57.6%).

Seed rotation intact -- training sees a different SUMO scenario every
episode (seed=42 for train, seed=0 for eval). PPO(seed=42) makes the
algorithm's own RNG (net init, rollout sampling) reproducible too.

For a DQN run to be a fair comparison against this one, see
train_dqn.py -- it must use identical SWITCH_PENALTY /
WASTED_VOTE_PENALTY / IMBALANCE_BONUS_WEIGHT / WRONG_DIRECTION_PENALTY /
MIN_GREEN, matched TOTAL_TIMESTEPS/gamma/seeds/eval protocol/network
width, and the same OUTPUT_FOLDER_NAME so the two output directories
are self-evidently a matched pair.
"""

import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.utils import get_linear_fn

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from env import SumoTrafficEnv2J


class EntropyAnnealCallback(BaseCallback):
    def __init__(self, start=0.05, end=0.01, total_timesteps=500_000, verbose=0):
        super().__init__(verbose)
        self.start = start
        self.end = end
        self.total_timesteps = total_timesteps

    def _on_step(self):
        frac = min(self.num_timesteps / self.total_timesteps, 1.0)
        self.model.ent_coef = self.start + frac * (self.end - self.start)
        return True


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Descriptive, self-documenting folder name -- encodes the hyperparameters
# that actually distinguish this run, so a folder name can never silently
# drift out of sync with the config that produced it. Keep this in sync
# with train_dqn.py's OUTPUT_FOLDER_NAME so it's always obvious which DQN
# folder a given PPO folder is meant to be compared against.
OUTPUT_FOLDER_NAME = "mg12_sw0.3_wd0.2"
MODELS_DIR = os.path.join(SCRIPT_DIR, "models", OUTPUT_FOLDER_NAME)

lr_schedule = get_linear_fn(start=3e-4, end=5e-5, end_fraction=1.0)

MAX_QUEUE = None
TOTAL_TIMESTEPS = 500_000

SWITCH_PENALTY = 0.3
WASTED_VOTE_PENALTY = 0.03
IMBALANCE_BONUS_WEIGHT = 0.0
WRONG_DIRECTION_PENALTY = 0.2
MIN_GREEN = 12


def make_train_env(seed, switch_penalty, wasted_vote_penalty, imbalance_bonus_weight,
                    wrong_direction_penalty, min_green, port):
    def _init():
        return SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            seed=seed,
            port=port,
            max_queue=MAX_QUEUE,
            switch_penalty=switch_penalty,
            wasted_vote_penalty=wasted_vote_penalty,
            imbalance_bonus_weight=imbalance_bonus_weight,
            wrong_direction_penalty=wrong_direction_penalty,
            min_green=min_green,
        )
    return _init


def make_eval_env(seed, switch_penalty, wasted_vote_penalty, imbalance_bonus_weight,
                   wrong_direction_penalty, min_green, port):
    def _init():
        return SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            seed=seed,
            port=port,
            max_queue=MAX_QUEUE,
            switch_penalty=switch_penalty,
            wasted_vote_penalty=wasted_vote_penalty,
            imbalance_bonus_weight=imbalance_bonus_weight,
            wrong_direction_penalty=wrong_direction_penalty,
            min_green=min_green,
        )
    return _init


def run_training(switch_penalty, wasted_vote_penalty, imbalance_bonus_weight,
                  wrong_direction_penalty, min_green, out_dir, train_port, eval_port):
    best_dir = os.path.join(out_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    train_env = make_vec_env(
        make_train_env(42, switch_penalty, wasted_vote_penalty,
                        imbalance_bonus_weight, wrong_direction_penalty, min_green, train_port),
        n_envs=1,
    )
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=False)

    eval_env = make_vec_env(
        make_eval_env(0, switch_penalty, wasted_vote_penalty,
                       imbalance_bonus_weight, wrong_direction_penalty, min_green, eval_port),
        n_envs=1,
    )
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        learning_rate=lr_schedule,
        n_steps=4096,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.75,
        max_grad_norm=0.5,
        target_kl=0.03,
        seed=42,
        verbose=1,
    )

    entropy_callback = EntropyAnnealCallback(start=0.05, end=0.01, total_timesteps=TOTAL_TIMESTEPS)
    callbacks = CallbackList([eval_callback, entropy_callback])

    print(f"\n{'='*70}")
    print(f"Training run: switch_penalty={switch_penalty}, "
          f"wasted_vote_penalty={wasted_vote_penalty}, "
          f"imbalance_bonus_weight={imbalance_bonus_weight}, "
          f"wrong_direction_penalty={wrong_direction_penalty}, "
          f"min_green={min_green}")
    print(f"Output -> {out_dir}")
    print(f"{'='*70}\n")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

    model.save(os.path.join(out_dir, "ppo_sumo_2junction"))
    train_env.save(os.path.join(out_dir, "vec_normalize_sumo.pkl"))

    train_env.close()
    eval_env.close()

    print(f"Training done -> {os.path.join(out_dir, 'ppo_sumo_2junction.zip')}")


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    run_training(
        switch_penalty=SWITCH_PENALTY,
        wasted_vote_penalty=WASTED_VOTE_PENALTY,
        imbalance_bonus_weight=IMBALANCE_BONUS_WEIGHT,
        wrong_direction_penalty=WRONG_DIRECTION_PENALTY,
        min_green=MIN_GREEN,
        out_dir=MODELS_DIR,
        train_port=8813,
        eval_port=8814,
    )