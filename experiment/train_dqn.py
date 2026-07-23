"""
train_dqn.py

DQN counterpart to train.py. Trains on the identical SumoTrafficEnv2J
(same reward config, same MIN_GREEN, same seed-rotation scheme, same
eval protocol) so the PPO-vs-DQN comparison is apples-to-apples on the
environment side. The only structural difference is the action-space
adapter (see wrappers.FlattenMultiDiscreteAction) -- SB3's DQN requires
a single Discrete action space, PPO doesn't.

CHANGED from the previous run (mg12_sw0.3_wd0.35, exploration_fraction=1.0):
that run showed directional agreement BELOW random chance at every
DirectionMonitor checkpoint from step 100k through 500k (J1/J2 both
sitting in the 32-50% band, no upward trend), mean_hold pinned at
12-13 (right at the MIN_GREEN=12 floor) the entire run, switch rate
near-ceiling (96-98% of physical max), and mean_q_spread=0.057 near the
end -- the network was barely differentiating actions by Q-value. That
combination (flat from early on, not slowly improving) points at
something structural, not "needs more steps":

  1. WRONG_DIRECTION_PENALTY reverted 0.35 -> 0.2, back to matching
     PPO's actual value. Raising it to 0.35 didn't fix directional
     agreement last time, so keeping it different from PPO now would
     only muddy attribution without evidence it helps.

  2. exploration_fraction lowered 1.0 -> 0.3. This was the actual
     suspect: it was originally set to 1.0 to "mirror" PPO's full-run
     entropy anneal, but that mirroring doesn't hold -- PPO's entropy
     bonus is a soft nudge (ent_coef added to the loss) that still lets
     the policy converge to near-deterministic behavior early if the
     advantage signal is strong. DQN's epsilon-greedy is a hard floor
     on random actions -- at exploration_fraction=1.0, a meaningful
     fraction of actions were still literally random deep into
     training, which is consistent with the near-zero Q-spread and the
     complete lack of drift in directional agreement over 500k steps.
     0.3 anneals epsilon to its floor by 150k steps, leaving 350k steps
     of near-greedy exploitation to actually commit to a policy --
     SB3's own DQN default is 0.1; 1.0 was likely an overcorrection for
     an algorithm that explores fundamentally differently than PPO.

Only these two changed, and only exploration_fraction is a real
hypothesis change -- WRONG_DIRECTION_PENALTY reverting to match PPO is
a control, not a fix, so if this run improves, the improvement is
attributable to the exploration schedule, not to reward tuning.

FAIRNESS NOTES -- read before writing these numbers into your report:

- Reward config, TOTAL_TIMESTEPS, gamma, seeds (train=42, eval=0),
  eval_freq/n_eval_episodes, network width [128, 128], and MIN_GREEN
  are matched to train.py's actual constants and passed explicitly to
  the env constructor rather than relied upon as class defaults:
  SWITCH_PENALTY=0.3, WASTED_VOTE_PENALTY=0.03,
  IMBALANCE_BONUS_WEIGHT=0.0, WRONG_DIRECTION_PENALTY=0.2, MIN_GREEN=12.

- exploration_fraction=0.3 is now the ONE deliberate difference from
  PPO's exploration schedule -- state this explicitly in your report
  if you compare hyperparameter tables side by side, since every other
  fairness note below still applies unchanged.

- model-level seed=42 added to the DQN(...) constructor, matching
  PPO's `seed=42`. This controls the algorithm's own internal RNG
  (network init, replay/exploration sampling) -- separate from the
  seed=42 passed to make_train_env, which controls SUMO scenario
  rotation and was already matched.

- DQN is off-policy and PPO is on-policy, so some hyperparameters have
  no PPO equivalent (buffer_size, target_update_interval) or share a
  name without being comparable 1:1. max_grad_norm is left at SB3's
  DQN default (10) rather than forced to match PPO's 0.5, because it
  bounds a TD-error gradient vs. a clipped surrogate-loss gradient --
  same number, different scale. Say this explicitly in your report if
  you list both hyperparameter tables side by side.

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

# CHANGED: folder name now encodes exploration_fraction too (ef0.3),
# distinguishing this run from BOTH the earlier mg12_sw0.3_wd0.2 (which
# never had this exploration fix) and the mg12_sw0.3_wd0.35 run (wrong
# variable changed) -- so all three DQN attempts stay on disk as
# distinct, comparable artifacts instead of overwriting each other.
OUTPUT_FOLDER_NAME = "mg12_sw0.3_wd0.2_ef0.3"
MODELS_DIR = os.path.join(SCRIPT_DIR, "models_dqn", OUTPUT_FOLDER_NAME)

lr_schedule = get_linear_fn(start=3e-4, end=5e-5, end_fraction=1.0)  # same schedule as PPO

MAX_QUEUE = None
TOTAL_TIMESTEPS = 500_000  # matched to train.py

# --- reward + dynamics config -- MATCHED to train.py's actual constants.
#     Don't change one without changing the other, or the comparison is
#     meaningless. ---
SWITCH_PENALTY = 0.3
WASTED_VOTE_PENALTY = 0.03
IMBALANCE_BONUS_WEIGHT = 0.0
WRONG_DIRECTION_PENALTY = 0.2  # CHANGED: reverted from 0.35 -- see module docstring
MIN_GREEN = 12  # matched to train.py; passed explicitly, not left to env.py's class default

# CHANGED: 1.0 -> 0.3 -- see module docstring for the full reasoning.
EXPLORATION_FRACTION = 0.3

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
            min_green=MIN_GREEN,
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
            min_green=MIN_GREEN,
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
        exploration_fraction=EXPLORATION_FRACTION,   # CHANGED: 1.0 -> 0.3, see module docstring
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        gamma=0.99,                                  # matched to PPO
        seed=42,                                     # matches PPO's seed=42
        verbose=1,
    )

    print(f"\n{'='*70}")
    print(f"DQN training run: switch_penalty={SWITCH_PENALTY}, "
          f"wasted_vote_penalty={WASTED_VOTE_PENALTY}, "
          f"imbalance_bonus_weight={IMBALANCE_BONUS_WEIGHT}, "
          f"wrong_direction_penalty={WRONG_DIRECTION_PENALTY}, "
          f"min_green={MIN_GREEN}, exploration_fraction={EXPLORATION_FRACTION}")
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