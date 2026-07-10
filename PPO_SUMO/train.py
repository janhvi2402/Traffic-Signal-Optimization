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
    `total_timesteps`. SB3's PPO does NOT support scheduling ent_coef
    the way it does learning_rate (ent_coef is a flat float arg, not a
    Schedule) — this callback works around that by mutating
    model.ent_coef directly at each training step.

    WHY: diagnostics on the previous model showed it had collapsed to
    voting action=1 every single step for both junctions, regardless
    of queue state (277/277 switches landing on the identical step,
    period exactly MIN_GREEN + YELLOW_TIME). That's a policy that
    stopped exploring before it ever discovered a genuinely reactive
    strategy. Starting entropy higher (0.05) keeps exploration alive
    longer early in training so alternative strategies actually get
    sampled, then decaying to 0.01 lets it converge to something sharp
    once it has found a better basin than "always switch".
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

# FIX: anchor all paths to this script's own folder, not the cwd —
# matches the pattern already used in test.py, so train.py and test.py
# always agree on where the model/normalizer live regardless of where
# you run each one from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  #location of current script
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
BEST_DIR   = os.path.join(MODELS_DIR, "best")

# Linear LR decay: 3e-4 -> 5e-5 over training
lr_schedule = get_linear_fn(start=3e-4, end=5e-5, end_fraction=1.0)

os.makedirs(BEST_DIR, exist_ok=True)  #if models does not exist it creates it,if already exist-nothing happens b/c exist_ok=True

# If you've calibrated MAX_QUEUE from a real rollout, set it here and
# it'll be passed into both train/eval envs so obs/reward normalisation
# stays consistent between training and evaluation.
# e.g. MAX_QUEUE = 10  (based on observed halting counts)
MAX_QUEUE = None          # None -> falls back to env's MAX_QUEUE_DEFAULT (30)
SWITCH_PENALTY = 0.15       # cost per agent-initiated switch; raised from 0.05
WASTED_VOTE_PENALTY = 0.03  # cost per ineligible action=1 vote — closes the
                             # "always vote 1 for free" loophole; see env.py docstring
TOTAL_TIMESTEPS = 500_000

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
            max_queue=MAX_QUEUE,
            switch_penalty=SWITCH_PENALTY,
            wasted_vote_penalty=WASTED_VOTE_PENALTY,
        )
    return _init

def make_eval_env(seed=0):
    def _init():
        return SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            seed=seed,
            port=8814,
            max_queue=MAX_QUEUE,
            switch_penalty=SWITCH_PENALTY,
            wasted_vote_penalty=WASTED_VOTE_PENALTY,
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
    eval_freq            = 20_000,   # steps between evaluations, evaluate model. save logs
    n_eval_episodes      = 3,
    deterministic        = True,
    verbose              = 1, #how much to print during training , verbose=0 print nothing, verbose =2 even more detailed debuging info
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
    # Initial value only — EntropyAnnealCallback overrides this every
    # step, decaying start=0.05 -> end=0.01 over training (see below).
    # Set here mainly so the very first rollout (before the callback's
    # first _on_step) still uses a sane value.
    ent_coef      = 0.05,
    vf_coef       = 0.75,          # was default 0.5 — give the critic more weight
    max_grad_norm = 0.5,           # add gradient clipping for stability
    target_kl     = 0.03,          # stop updates early if policy shifts too much in one epoch
    verbose       = 1,
)

entropy_callback = EntropyAnnealCallback(
    start=0.05, end=0.01, total_timesteps=TOTAL_TIMESTEPS
)
callbacks = CallbackList([eval_callback, entropy_callback])

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

model.save(os.path.join(MODELS_DIR, "ppo_sumo_2junction"))
train_env.save(os.path.join(MODELS_DIR, "vec_normalize_sumo.pkl"))
print(f"Training done → {os.path.join(MODELS_DIR, 'ppo_sumo_2junction.zip')}")