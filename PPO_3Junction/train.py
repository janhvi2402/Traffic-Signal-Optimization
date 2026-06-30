from PPO_3Junction.env import TrafficEnv3J
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
import functools

# randomize_rates=True for training — different traffic every episode
train_env = make_vec_env(
    TrafficEnv3J,
    n_envs=4,
    env_kwargs={"randomize_rates": True}   # passed to TrafficEnv3J() TrafficEnv3J(**env_kwargs), argument to it
)
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

# fixed rates for eval so you can track progress consistently
eval_env = make_vec_env(
    TrafficEnv3J,
    n_envs=1,
    env_kwargs={"randomize_rates": False}  # fixed default rates during eval
)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best/",
    eval_freq=5000,
    n_eval_episodes=10,
    verbose=1,
)

model = PPO(
    policy="MlpPolicy",
    env=train_env,
    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    learning_rate=1e-4,
    n_steps=4096,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,
    verbose=1,
)

model.learn(total_timesteps=2_000_000, callback=eval_callback)
model.save("models/ppo_3junction")
train_env.save("models/vec_normalize.pkl")
print("Training done. Model saved to models/ppo_3junction.zip")