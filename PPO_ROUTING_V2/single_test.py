import os
from stable_baselines3 import PPO
from single_env import SumoSingleJunctionEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "ppo_single_junction")

model = PPO.load(MODEL_PATH)

env = SumoSingleJunctionEnv(
    use_gui=True,
    seed=0,
    randomize_routes=True,
    time_dropout_prob=0.0,   # IMPORTANT: no dropout during evaluation
)

obs, _ = env.reset()

done = False
step = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)

    print(
        f"Step {step:4d} | "
        f"NS={obs[0]:.2f} EW={obs[1]:.2f} "
        f"Imb={obs[7]:.2f} "
        f"Action={int(action)}"
    )

    obs, reward, done, _, _ = env.step(action)
    step += 1

env.close()