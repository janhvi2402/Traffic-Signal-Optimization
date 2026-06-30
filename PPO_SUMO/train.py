import os
import functools
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from sumo_env import SumoTrafficEnv2J

os.makedirs("models/best", exist_ok=True)

# ── environment factories ──────────────────────────────────────────────────────
# SUMO can't run two instances on the same port, so each parallel env
# gets its own seed (which randomises vehicle insertions) rather than
# sharing state.  n_envs=1 is safe; increase only if you launch each
# env on a separate TraCI port (see sumo --remote-port).

def make_env(seed=0):
    def _init():
        env = SumoTrafficEnv2J(
            cfg_path  = os.path.join(os.path.dirname(__file__), "network.sumocfg"),
            use_gui   = False,
            max_steps = 3600,
            seed      = seed,
        )
        return env
    return _init

# Single training env (SUMO only supports 1 TraCI connection per process
# without extra port configuration).
train_env = make_vec_env(make_env(seed=42), n_envs=1)
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

eval_env  = make_vec_env(make_env(seed=0), n_envs=1)
eval_env  = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

# ── callbacks ─────────────────────────────────────────────────────────────────
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path = "./models/best/",
    eval_freq            = 10_000,   # steps between evaluations
    n_eval_episodes      = 3,
    deterministic        = True,
    verbose              = 1,
)

# ── model ─────────────────────────────────────────────────────────────────────
model = PPO(
    policy       = "MlpPolicy",
    env          = train_env,
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    learning_rate = 3e-4,
    n_steps       = 2048,   # collected per update (1 env × 2048 steps)
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    verbose       = 1,
)

model.learn(total_timesteps=500_000, callback=eval_callback)

model.save("models/ppo_sumo_2junction")
train_env.save("models/vec_normalize_sumo.pkl")
print("Training done → models/ppo_sumo_2junction.zip")