import os
from env import SumoTrafficEnv2J

env = SumoTrafficEnv2J(
    cfg_path=os.path.join(os.path.dirname(__file__), "network.sumocfg"),
    use_gui=False,
    max_steps=200,
    seed=0,
)
#no need for this file
obs, _ = env.reset()
for step in range(200):
    obs, r, done, _, _ = env.step([0, 0])  # never request switch
    print(step, env.conn.trafficlight.getPhase("J1"), env.conn.trafficlight.getPhase("J2"))
    if done:
        break

env.close()