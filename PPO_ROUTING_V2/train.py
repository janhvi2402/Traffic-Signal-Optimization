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

TOTAL_TIMESTEPS = 500_000

# reward config -- matches the values that measurably improved the
# centralized 2-junction run (SWITCH_PENALTY 0.3->0.4,
# WRONG_DIRECTION_PENALTY 0.15->0.25, MIN_GREEN 10->15 is set directly
# in single_env.py's class constant, not overridden here)
SWITCH_PENALTY          = 0.4
WASTED_VOTE_PENALTY     = 0.03
IMBALANCE_BONUS_WEIGHT  = 0.0
WRONG_DIRECTION_PENALTY = 0.25


class EntropyAnnealCallback(BaseCallback):
    def __init__(self, start=0.02, end=0.01, total_timesteps=TOTAL_TIMESTEPS, verbose=0):
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
    entropy_callback = EntropyAnnealCallback(start=0.02, end=0.01, total_timesteps=TOTAL_TIMESTEPS)
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
        ent_coef=0.02,   # overridden step-by-step by EntropyAnnealCallback
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        seed=42,
        verbose=1,
    )

    print(f"\n{'='*70}")
    print(f"Training run: switch_penalty={SWITCH_PENALTY}, "
          f"wasted_vote_penalty={WASTED_VOTE_PENALTY}, "
          f"imbalance_bonus_weight={IMBALANCE_BONUS_WEIGHT}, "
          f"wrong_direction_penalty={WRONG_DIRECTION_PENALTY}, "
          f"MIN_GREEN={SumoSingleJunctionEnv.MIN_GREEN}")
    print(f"Output -> {MODELS_DIR}")
    print(f"{'='*70}\n")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

    model.save(os.path.join(MODELS_DIR, "ppo_single_junction"))
    train_env.close()
    eval_env.close()

    print(f"Training done -> {os.path.join(MODELS_DIR, 'ppo_single_junction.zip')}")