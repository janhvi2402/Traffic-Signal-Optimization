"""
train_dqn.py

DQN counterpart to train.py. Trains on the identical SumoTrafficEnv2J
(same reward config, same seed-rotation scheme, same eval protocol) so
the eventual PPO-vs-DQN comparison is apples-to-apples on the
environment side. The only structural difference is the action-space
adapter (see wrappers.FlattenMultiDiscreteAction) -- SB3's DQN requires
a single Discrete action space, PPO doesn't.

FAIRNESS NOTES -- read before writing these numbers into your report:

- Reward config, TOTAL_TIMESTEPS, gamma, seeds (train=42, eval=0),
  eval_freq/n_eval_episodes, and network width [128, 128] are matched
  to train.py's current run (SWITCH_PENALTY=0.3, WRONG_DIRECTION_PENALTY
  =0.15) exactly. If you change train.py's reward config before
  retraining PPO, mirror the change here too or the comparison breaks.

- DQN is off-policy and PPO is on-policy, so some hyperparameters have
  no PPO equivalent (buffer_size, target_update_interval, exploration
  schedule) or share a name without being comparable 1:1. In particular
  max_grad_norm is left at SB3's DQN default (10) rather than forced to
  match PPO's 0.5, because it bounds a TD-error gradient vs. a clipped
  surrogate-loss gradient -- same number, different scale. Say this
  explicitly in your report if you list both hyperparameter tables
  side by side; don't imply they were tuned to be equal.

- exploration_fraction=1.0 anneals epsilon across the *entire* run,
  mirroring how EntropyAnnealCallback anneals PPO's ent_coef across the
  entire run in train.py. This keeps the "explore early, exploit late"
  shape comparable instead of one algorithm exploring for 10% of
  training and the other for 100%.

- Ports (8823/8824) are offset from train.py's (8813/8814) so you can
  run a PPO sweep and this DQN run in separate terminals at the same
  time without a TraCI port collision.
"""

import os
import sys
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import get_linear_fn
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))

from env import SumoTrafficEnv2J
from wrappers import FlattenMultiDiscreteAction

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Mirrors train.py's OUTPUT_FOLDER_NAME pattern: this run trains against
# whatever env.py CURRENTLY defines (16-dim obs, explicit imbalance
# feature). The wrapper only touches the action space, so the 16-dim
# observation passes straight through with no code change here -- but
# the folder name matters: if env.py's observation space changes again
# later, a model saved under this folder becomes stale/incompatible,
# same as models/obs_imbalance_feature/ would be for PPO. Keep this in
# sync with train.py's OUTPUT_FOLDER_NAME so it's always obvious which
# PPO folder a given DQN folder is meant to be compared against.
OUTPUT_FOLDER_NAME = "obs_imbalance_feature"
MODELS_DIR = os.path.join(SCRIPT_DIR, "models_dqn", OUTPUT_FOLDER_NAME)

lr_schedule = get_linear_fn(start=3e-4, end=5e-5, end_fraction=1.0)  # same schedule as PPO

MAX_QUEUE = None
TOTAL_TIMESTEPS = 500_000  # matched to train.py

# --- reward config -- MATCHED to train.py's current run. Don't change
#     one without changing the other, or the comparison is meaningless. ---
SWITCH_PENALTY = 0.3
WASTED_VOTE_PENALTY = 0.03
IMBALANCE_BONUS_WEIGHT = 0.0
WRONG_DIRECTION_PENALTY = 0.15

TRAIN_PORT = 8823
EVAL_PORT = 8824


def make_train_env(seed, port):
    def _init():
        env = SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            seed=seed,
            port=port,
            max_queue=MAX_QUEUE,
            switch_penalty=SWITCH_PENALTY,
            wasted_vote_penalty=WASTED_VOTE_PENALTY,
            imbalance_bonus_weight=IMBALANCE_BONUS_WEIGHT,
            wrong_direction_penalty=WRONG_DIRECTION_PENALTY,
        )
        return FlattenMultiDiscreteAction(env)
    return _init


def make_eval_env(seed, port):
    def _init():
        env = SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            seed=seed,
            port=port,
            max_queue=MAX_QUEUE,
            switch_penalty=SWITCH_PENALTY,
            wasted_vote_penalty=WASTED_VOTE_PENALTY,
            imbalance_bonus_weight=IMBALANCE_BONUS_WEIGHT,
            wrong_direction_penalty=WRONG_DIRECTION_PENALTY,
        )
        return FlattenMultiDiscreteAction(env)
    return _init


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_dir = os.path.join(MODELS_DIR, "best")
    os.makedirs(best_dir, exist_ok=True)

    train_env = make_vec_env(make_train_env(42, TRAIN_PORT), n_envs=1)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=False)

    eval_env = make_vec_env(make_eval_env(0, EVAL_PORT), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        eval_freq=20_000,      # matched to train.py
        n_eval_episodes=5,     # matched to train.py
        deterministic=True,
        verbose=1,
    )

    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        policy_kwargs=dict(net_arch=[128, 128]),   # same width as PPO's pi/vf nets
        learning_rate=lr_schedule,                  # same schedule as PPO
        buffer_size=100_000,                        # ~28 episodes of replay (3600 steps/ep)
        learning_starts=5_000,
        batch_size=128,                             # matched to PPO's batch_size
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=1.0,                   # anneal eps across the WHOLE run,
                                                       # mirroring EntropyAnnealCallback's
                                                       # full-run anneal for PPO
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        gamma=0.99,                                  # matched to PPO
        verbose=1,
    )

    print(f"\n{'='*70}")
    print(f"DQN training run: switch_penalty={SWITCH_PENALTY}, "
          f"wasted_vote_penalty={WASTED_VOTE_PENALTY}, "
          f"imbalance_bonus_weight={IMBALANCE_BONUS_WEIGHT}, "
          f"wrong_direction_penalty={WRONG_DIRECTION_PENALTY}")
    print(f"Output -> {MODELS_DIR}")
    print(f"{'='*70}\n")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

    model.save(os.path.join(MODELS_DIR, "dqn_sumo_2junction"))
    train_env.save(os.path.join(MODELS_DIR, "vec_normalize_sumo_dqn.pkl"))

    train_env.close()
    eval_env.close()

    print(f"Training done -> {os.path.join(MODELS_DIR, 'dqn_sumo_2junction.zip')}")


if __name__ == "__main__":
    main()