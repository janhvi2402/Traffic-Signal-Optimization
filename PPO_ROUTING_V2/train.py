import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import get_linear_fn

from single_env import SumoSingleJunctionEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
BEST_DIR   = os.path.join(MODELS_DIR, "best")
os.makedirs(BEST_DIR, exist_ok=True)

lr_schedule = get_linear_fn(start=3e-4, end=5e-5, end_fraction=1.0)


def make_train_env(seed=0):
    def _init():
        return SumoSingleJunctionEnv(
            seed=seed,
            port=8815,
            randomize_routes=True,   # domain randomization -- the core fix
            # time_dropout_prob removed: it collided with the legitimate
            # 0.0 value at real phase-start and just added label noise
            # instead of cleanly ablating the timing shortcut. The
            # directional switch bonus + wasted-vote penalty in
            # single_env.py now do the real work of discouraging
            # cadence-based switching.
        )
    return _init


def make_eval_env(seed=0):
    def _init():
        # eval env: keep randomized routes so eval reflects deployment
        # conditions
        return SumoSingleJunctionEnv(
            seed=seed,
            port=8816,
            randomize_routes=True,
        )
    return _init


train_env = make_vec_env(make_train_env(seed=42), n_envs=1)
eval_env  = make_vec_env(make_eval_env(seed=0), n_envs=1)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path = BEST_DIR,
    eval_freq            = 20_000,
    n_eval_episodes       = 5,   # bumped from 3 -- more episodes needed
                                  # to get a stable read with randomized
                                  # demand across episodes
    deterministic         = True,
    verbose               = 1,
)

model = PPO(
    policy        = "MlpPolicy",
    env           = train_env,
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
    learning_rate = lr_schedule,
    n_steps       = 2048,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    target_kl     = 0.03,
    verbose       = 1,
)

# more timesteps than before -- randomized demand is a harder training
# distribution than a fixed one, so it needs more samples to converge
model.learn(total_timesteps=500_000, callback=eval_callback)

model.save(os.path.join(MODELS_DIR, "ppo_single_junction"))
print(f"Training done -> {os.path.join(MODELS_DIR, 'ppo_single_junction.zip')}")