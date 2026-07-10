import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env import SumoTrafficEnv2J

# Same anchoring pattern as train.py — run this from anywhere and it
# still finds the right model/env files.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

# Point this at whichever checkpoint you want to inspect —
# the final save, or models/best/best_model.zip from EvalCallback.
MODEL_PATH = os.path.join(MODELS_DIR, "ppo_sumo_2junction")
# MODEL_PATH = os.path.join(MODELS_DIR, "best", "best_model")

# Must match whatever you trained with, or obs/reward normalisation
# will be inconsistent with what the policy learned on.
MAX_QUEUE = None
SWITCH_PENALTY = 0.05

EPISODE_LEN = 3600   # steps to run; set lower for a quicker look


def run_diagnostic(use_gui=False):
    env = SumoTrafficEnv2J(
        cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
        seed=0,
        port=8815,              # different port from train/eval envs so
                                 # you can run this alongside training if needed
        max_queue=MAX_QUEUE,
        switch_penalty=SWITCH_PENALTY,
        use_gui=use_gui,
        max_steps=EPISODE_LEN,
    )

    model = PPO.load(MODEL_PATH)

    obs, _ = env.reset()
    history = {"J1_q": [], "J2_q": [], "J1_sw": [], "J2_sw": []}

    for _ in range(EPISODE_LEN):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        history["J1_q"].append(info["local_queue"]["J1"])
        history["J2_q"].append(info["local_queue"]["J2"])
        history["J1_sw"].append(info["switched"]["J1"])
        history["J2_sw"].append(info["switched"]["J2"])
        if done:
            break

    env.close()
    return history


def plot_history(history, out_path):
    steps = range(len(history["J1_q"]))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(steps, history["J1_q"], label="J1 queue", color="tab:blue")
    ax1.plot(steps, history["J2_q"], label="J2 queue", color="tab:orange")
    ax1.set_ylabel("mean queue (veh)")
    ax1.legend()
    ax1.set_title("Per-junction queue over the episode")

    # Mark switch events as vertical ticks on a 0/1 line per junction,
    # offset slightly so they don't overlap when both fire together.
    j1_sw_steps = [i for i, s in enumerate(history["J1_sw"]) if s]
    j2_sw_steps = [i for i, s in enumerate(history["J2_sw"]) if s]
    ax2.eventplot(j1_sw_steps, lineoffsets=1, color="tab:blue", label="J1 switch")
    ax2.eventplot(j2_sw_steps, lineoffsets=0, color="tab:orange", label="J2 switch")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["J2", "J1"])
    ax2.set_xlabel("step")
    ax2.set_title("Switch events per junction")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved diagnostic plot -> {out_path}")


if __name__ == "__main__":
    history = run_diagnostic(use_gui=False)

    n_j1_sw = sum(history["J1_sw"])
    n_j2_sw = sum(history["J2_sw"])
    both_same_step = sum(
        1 for a, b in zip(history["J1_sw"], history["J2_sw"]) if a and b
    )
    print(f"J1 switches: {n_j1_sw}  |  J2 switches: {n_j2_sw}")
    print(f"Switches on the SAME step: {both_same_step} "
          f"({100 * both_same_step / max(n_j1_sw, n_j2_sw, 1):.1f}% overlap "
          f"of the more active junction)")

    out_path = os.path.join(SCRIPT_DIR, "diagnostic_plot.png")
    plot_history(history, out_path)