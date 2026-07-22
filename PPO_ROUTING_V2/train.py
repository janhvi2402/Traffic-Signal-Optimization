import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.utils import get_linear_fn

from single_env import SumoSingleJunctionEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
BEST_DIR   = os.path.join(MODELS_DIR, "best")
os.makedirs(BEST_DIR, exist_ok=True)

lr_schedule = get_linear_fn(start=3e-4, end=5e-5, end_fraction=1.0)

# raised 750k -> 1,000,000: this is the last training run before
# deployment, and the reward change below (WASTED_VOTE_PENALTY) is a
# meaningful shift in the incentive landscape -- more updates gives PPO
# more room to settle into genuinely selective voting instead of
# collapsing toward a shortcut again
TOTAL_TIMESTEPS = 1_000_000

# Reward config -- FINAL correction after diagnosing a policy collapse:
# raw action logs from the IMBALANCE_BONUS_WEIGHT=1.5 run showed
# model.predict() returning action=1 on essentially every printed step,
# across queue values from 0.00-0.40 and imbalance 0.32-0.70 -- the
# observation had stopped influencing the vote. Confirmed numerically:
# diagnostic.py's mean hold duration (17.4) was barely above the mean
# SAMPLED MIN_GREEN (16.4) -- ~1 step of real slack, meaning switches
# were firing the instant they became legally eligible, not when queue
# state actually warranted it. The previously-reported positive hold/
# imbalance correlation was a CONFOUND of the randomized MIN_GREEN
# itself (longer random draws mechanically give more time for imbalance
# to build before a still-timer-triggered switch), not evidence of
# learning.
#
# ROOT CAUSE: WASTED_VOTE_PENALTY=0.02 was too cheap. Voting 1 through
# ~15 ineligible steps cost only ~0.3 total per phase -- with MIN_GREEN
# hidden from the observation, "always vote yes" became a cheap,
# low-variance hedge that guaranteed catching the earliest legal switch
# every phase, regardless of that episode's actual MIN_GREEN. This
# defeated the entire purpose of IMBALANCE_BONUS_WEIGHT.
#
# FIX: WASTED_VOTE_PENALTY raised 0.02 -> 0.1 (now comparable to
# SWITCH_PENALTY). Blanket voting through ~15 ineligible steps now costs
# ~1.5 per phase -- a large fraction of that phase's total reward,
# making it clearly worse than withholding the vote until warranted.
# This is NOT expected to flip to the opposite collapse ("always vote 0
# until forced at MAX_GREEN=90"): IMBALANCE_BONUS_WEIGHT=1.5 already
# bleeds substantial negative reward every step spent on the wrong side,
# so staying wrong for up to 90 forced steps is worse than an occasional
# wasted vote. See single_env.py's docstring for the full reasoning.
SWITCH_PENALTY          = 0.1
WASTED_VOTE_PENALTY     = 0.1     # was 0.02 -- the actual fix this round
IMBALANCE_BONUS_WEIGHT  = 1.5
WRONG_DIRECTION_PENALTY = 0.2


class EntropyAnnealCallback(BaseCallback):
    def __init__(self, start=0.04, end=0.02, total_timesteps=TOTAL_TIMESTEPS, verbose=0):
        super().__init__(verbose)
        self.start = start
        self.end = end
        self.total_timesteps = total_timesteps

    def _on_step(self):
        frac = min(self.num_timesteps / self.total_timesteps, 1.0)
        self.model.ent_coef = self.start + frac * (self.end - self.start)
        return True


def make_train_env(seed=42):
    def _init():
        return SumoSingleJunctionEnv(
            seed=seed,
            port=8815,
            randomize_routes=True,
            switch_penalty=SWITCH_PENALTY,
            wasted_vote_penalty=WASTED_VOTE_PENALTY,
            imbalance_bonus_weight=IMBALANCE_BONUS_WEIGHT,
            wrong_direction_penalty=WRONG_DIRECTION_PENALTY,
        )
    return _init


def make_eval_env(seed=0):
    def _init():
        return SumoSingleJunctionEnv(
            seed=seed,
            port=8816,
            randomize_routes=True,
            switch_penalty=SWITCH_PENALTY,
            wasted_vote_penalty=WASTED_VOTE_PENALTY,
            imbalance_bonus_weight=IMBALANCE_BONUS_WEIGHT,
            wrong_direction_penalty=WRONG_DIRECTION_PENALTY,
        )
    return _init


if __name__ == "__main__":
    train_env = make_vec_env(make_train_env(seed=42), n_envs=1)
    eval_env  = make_vec_env(make_eval_env(seed=0), n_envs=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_DIR,
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    # entropy floor raised 0.01 -> 0.02 (and start 0.03 -> 0.04): a
    # near-constant action is a classic entropy-collapse failure mode --
    # once the categorical distribution over {0,1} saturates toward one
    # side, gradients through log-prob shrink and it tends to stay
    # stuck. Keeping a higher entropy floor for the whole run (not just
    # early on) makes it harder for the policy to fully collapse again.
    entropy_callback = EntropyAnnealCallback(start=0.04, end=0.02, total_timesteps=TOTAL_TIMESTEPS)
    callbacks = CallbackList([eval_callback, entropy_callback])

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        learning_rate=lr_schedule,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.04,   # overridden step-by-step by EntropyAnnealCallback
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        seed=42,
        verbose=1,
    )

    print(f"\n{'='*70}")
    print(f"FINAL training run: switch_penalty={SWITCH_PENALTY}, "
          f"wasted_vote_penalty={WASTED_VOTE_PENALTY}, "
          f"imbalance_bonus_weight={IMBALANCE_BONUS_WEIGHT}, "
          f"wrong_direction_penalty={WRONG_DIRECTION_PENALTY}, "
          f"MIN_GREEN_RANGE={SumoSingleJunctionEnv.MIN_GREEN_RANGE}, "
          f"total_timesteps={TOTAL_TIMESTEPS}")
    print(f"Output -> {MODELS_DIR}")
    print("NOTE: retrained from scratch -- do not warm-start from a checkpoint")
    print("trained under a previous reward config, its value function won't transfer.")
    print("After training, run diagnostic.py FIRST and check the new")
    print("'wasted-vote rate among votes cast' and 'slack' lines before anything else --")
    print("that's the direct check for whether this run's fix actually worked.")
    print(f"{'='*70}\n")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

    model.save(os.path.join(MODELS_DIR, "ppo_single_junction"))
    train_env.close()
    eval_env.close()

    print(f"Training done -> {os.path.join(MODELS_DIR, 'ppo_single_junction.zip')}")