import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import get_linear_fn

from env import SumoTrafficEnv2J

# FIX: anchor all paths to this script's own folder, not the cwd —
# matches the pattern already used in test.py, so train.py and test.py
# always agree on where the model/normalizer live regardless of where
# you run each one from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
BEST_DIR   = os.path.join(MODELS_DIR, "best")

# Linear LR decay: 3e-4 -> 5e-5 over training
lr_schedule = get_linear_fn(start=3e-4, end=5e-5, end_fraction=1.0)

os.makedirs(BEST_DIR, exist_ok=True)

# environment factories 
# SUMO can't run two instances on the same port, so each parallel env
# gets its own seed (which randomises vehicle insertions) rather than
# sharing state.  n_envs=1 is safe; increase only if you launch each
# env on a separate TraCI port (see sumo --remote-port).

def make_train_env(seed=0):
    def _init():
        return SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            seed=seed,
            port=8813,
        )
    return _init

def make_eval_env(seed=0):
    def _init():
        return SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            seed=seed,
            port=8814,
        )
    return _init

train_env = make_vec_env(make_train_env(seed=42), n_envs=1)
train_env = VecNormalize(train_env, norm_obs=False, norm_reward=False)  # was True, True

eval_env  = make_vec_env(make_eval_env(seed=0), n_envs=1)
eval_env  = VecNormalize(eval_env, norm_obs=False, norm_reward=False)  # was norm_obs=True

# callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path = BEST_DIR,
    eval_freq            = 20_000,   # steps between evaluations
    n_eval_episodes      = 3,
    deterministic        = True,
    verbose              = 1,
)

# model
model = PPO(
    policy        = "MlpPolicy",
    env           = train_env,
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    learning_rate = lr_schedule,   # was flat 3e-4
    n_steps       = 4096,          # was 2048 — more data per update, less noisy value target
    batch_size    = 128,           # was 64 — bigger minibatches, smoother gradients
    n_epochs      = 4,             # was 10 — fewer passes over noisy data = less overfitting
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    vf_coef       = 0.75,          # was default 0.5 — give the critic more weight
    max_grad_norm = 0.5,           # add gradient clipping for stability
    target_kl     = 0.03,          # stop updates early if policy shifts too much in one epoch
    verbose       = 1,
)

model.learn(total_timesteps=500_000, callback=eval_callback)

model.save(os.path.join(MODELS_DIR, "ppo_sumo_2junction"))
train_env.save(os.path.join(MODELS_DIR, "vec_normalize_sumo.pkl"))
print(f"Training done → {os.path.join(MODELS_DIR, 'ppo_sumo_2junction.zip')}")