"""
train.py

Now supports:
  - imbalance_bonus_weight passed through to the env (default 0.0,
    same as before unless you set IMBALANCE_BONUS_WEIGHT > 0)
  - a SWITCH_PENALTY sweep: set SWEEP_SWITCH_PENALTIES to a list of
    values and this will train one model per value, each saved to
    its own models/sweep_<value>/ folder, so you can compare configs
    side by side with test_diagnostic_imbalance.py afterward.

Set SWEEP_SWITCH_PENALTIES = None to fall back to a single run using
SWITCH_PENALTY, same as before.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.utils import get_linear_fn

from env import SumoTrafficEnv2J


class EntropyAnnealCallback(BaseCallback):
    """
    Linearly anneal ent_coef from `start` down to `end` over
    `total_timesteps`. See original docstring — unchanged.
    """
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
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

lr_schedule = get_linear_fn(start=3e-4, end=5e-5, end_fraction=1.0)

MAX_QUEUE = None
TOTAL_TIMESTEPS = 500_000

# --- base config (used when SWEEP_SWITCH_PENALTIES is None) ---
SWITCH_PENALTY = 0.15
WASTED_VOTE_PENALTY = 0.03
IMBALANCE_BONUS_WEIGHT = 0.0   # NEW — set >0 (e.g. 0.03) to reward
                                # holding green on the busier side

# --- sweep config ---
# Set to None for a single run using the constants above.
# Set to a list to train one model per SWITCH_PENALTY value, each
# combined with WASTED_VOTE_PENALTY and IMBALANCE_BONUS_WEIGHT below.
SWEEP_SWITCH_PENALTIES = [0.15, 0.3, 0.5]     # NEW
SWEEP_IMBALANCE_BONUS_WEIGHT = 0.03            # NEW — used only in sweep runs


def make_train_env(seed, switch_penalty, wasted_vote_penalty, imbalance_bonus_weight, port):
    def _init():
        return SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            seed=seed,
            port=port,
            max_queue=MAX_QUEUE,
            switch_penalty=switch_penalty,
            wasted_vote_penalty=wasted_vote_penalty,
            imbalance_bonus_weight=imbalance_bonus_weight,   # NEW
        )
    return _init


def make_eval_env(seed, switch_penalty, wasted_vote_penalty, imbalance_bonus_weight, port):
    def _init():
        return SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            seed=seed,
            port=port,
            max_queue=MAX_QUEUE,
            switch_penalty=switch_penalty,
            wasted_vote_penalty=wasted_vote_penalty,
            imbalance_bonus_weight=imbalance_bonus_weight,   # NEW
        )
    return _init


def run_training(
    switch_penalty,
    wasted_vote_penalty,
    imbalance_bonus_weight,
    out_dir,
    train_port,
    eval_port,
):
    """One full training run with a given reward config, saved to out_dir."""
    best_dir = os.path.join(out_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    train_env = make_vec_env(
        make_train_env(42, switch_penalty, wasted_vote_penalty, imbalance_bonus_weight, train_port),
        n_envs=1,
    )
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=False)

    eval_env = make_vec_env(
        make_eval_env(0, switch_penalty, wasted_vote_penalty, imbalance_bonus_weight, eval_port),
        n_envs=1,
    )
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        eval_freq=20_000,
        n_eval_episodes=3,
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
        verbose=1,
    )

    entropy_callback = EntropyAnnealCallback(
        start=0.05, end=0.01, total_timesteps=TOTAL_TIMESTEPS
    )
    callbacks = CallbackList([eval_callback, entropy_callback])

    print(f"\n{'='*70}")
    print(f"Training run: switch_penalty={switch_penalty}, "
          f"wasted_vote_penalty={wasted_vote_penalty}, "
          f"imbalance_bonus_weight={imbalance_bonus_weight}")
    print(f"Output -> {out_dir}")
    print(f"{'='*70}\n")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

    model.save(os.path.join(out_dir, "ppo_sumo_2junction"))
    train_env.save(os.path.join(out_dir, "vec_normalize_sumo.pkl"))

    train_env.close()
    eval_env.close()

    print(f"Training done -> {os.path.join(out_dir, 'ppo_sumo_2junction.zip')}")


if __name__ == "__main__":
    if SWEEP_SWITCH_PENALTIES is None:
        # single run, original behavior
        os.makedirs(MODELS_DIR, exist_ok=True)
        run_training(
            switch_penalty=SWITCH_PENALTY,
            wasted_vote_penalty=WASTED_VOTE_PENALTY,
            imbalance_bonus_weight=IMBALANCE_BONUS_WEIGHT,
            out_dir=MODELS_DIR,
            train_port=8813,
            eval_port=8814,
        )
    else:
        # sweep: one run per SWITCH_PENALTY value, own ports so they
        # don't collide if you ever parallelize this later, and own
        # output folder so test_diagnostic_imbalance.py can point at
        # each independently.
        base_port = 8820
        for i, sp in enumerate(SWEEP_SWITCH_PENALTIES):
            out_dir = os.path.join(MODELS_DIR, f"sweep_sp{str(sp).replace('.', '')}")
            os.makedirs(out_dir, exist_ok=True)
            run_training(
                switch_penalty=sp,
                wasted_vote_penalty=WASTED_VOTE_PENALTY,
                imbalance_bonus_weight=SWEEP_IMBALANCE_BONUS_WEIGHT,
                out_dir=out_dir,
                train_port=base_port + i * 2,
                eval_port=base_port + i * 2 + 1,
            )