from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
from env import TrafficEnv

# 1. create first
train_env = make_vec_env(TrafficEnv, n_envs=4)
eval_env  = make_vec_env(TrafficEnv, n_envs=1)

# 2. then wrap
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
eval_env  = VecNormalize(eval_env,  norm_obs=True, norm_reward=False)

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
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,
    verbose=1,
)

model.learn(total_timesteps=1_000_000, callback=eval_callback)
model.save("models/ppo_traffic")
train_env.save("models/vec_normalize.pkl")
print("Training done. Model saved to models/ppo_traffic.zip")