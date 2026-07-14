"""
Diagnostic eval for the single-junction PPO policy.

Beyond just running the policy, this logs, for every step where a switch
was actually voted for AND takes effect:
    - time_in_phase at the moment of the vote (was the vote basically on
      a fixed cadence, i.e. always ~MIN_GREEN?)
    - signed imbalance at the moment of the vote (did the vote track
      which side was actually heavier?)

At the end it prints:
    - the correlation between time_in_phase-at-switch and its variance
      (low variance / clustering near MIN_GREEN => cadence-driven)
    - the fraction of switches that were "correct" (toward the heavier
      side), using SumoSingleJunctionEnv.last_switch_was_correct
    - the correlation between switch votes and imbalance magnitude

If switches cluster tightly around MIN_GREEN regardless of imbalance
sign, the policy is still keying off elapsed time. If "correct switch"
fraction is well above 50% and switch votes correlate with |imbalance|,
the fix worked.
"""
import os
import numpy as np
from stable_baselines3 import PPO
from single_env import SumoSingleJunctionEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "ppo_single_junction")

model = PPO.load(MODEL_PATH)

env = SumoSingleJunctionEnv(
    use_gui=True,
    seed=0,
    randomize_routes=True,
)

obs, _ = env.reset()

done = False
step = 0

# per-switch log: (time_in_phase_at_vote, signed_imbalance_at_vote, correct)
switch_log = []
action_log = []  # (imbalance_signed, action) for every step, for correlation

while not done:
    action, _ = model.predict(obs, deterministic=True)
    action_int = int(action)

    # signed imbalance from the raw obs (undo the [0,1] rescale)
    signed_imbalance = obs[7] * 2.0 - 1.0
    time_in_phase = obs[5]

    print(
        f"Step {step:4d} | "
        f"NS={obs[0]:.2f} EW={obs[1]:.2f} "
        f"Imb={signed_imbalance:+.2f} "
        f"TimeInPhase={time_in_phase:.2f} "
        f"Action={action_int}"
    )

    action_log.append((signed_imbalance, action_int))

    was_in_yellow_before = bool(obs[6])
    obs, reward, done, _, _ = env.step(action)
    step += 1

    # a switch "landed" this step if we just transitioned into the new
    # green phase (env sets last_switch_was_correct at that moment)
    if env.last_switch_was_correct is not None:
        switch_log.append((time_in_phase, signed_imbalance, env.last_switch_was_correct))
        env.last_switch_was_correct = None  # consume it

env.close()

# --- summary ---
print("\n" + "=" * 60)
print(f"Total switches this episode: {len(switch_log)}")

if switch_log:
    times = np.array([t for t, _, _ in switch_log])
    imbs  = np.array([i for _, i, _ in switch_log])
    correct = np.array([c for _, _, c in switch_log])

    print(f"time_in_phase at switch request: mean={times.mean():.2f}, std={times.std():.2f}")
    print(f"  -> low std clustered near MIN_GREEN suggests cadence-driven switching")
    print(f"|imbalance| at switch request:   mean={np.abs(imbs).mean():.2f}, std={np.abs(imbs).std():.2f}")
    print(f"Fraction of switches toward the heavier side (correct): {correct.mean():.2%}")
    print(f"  -> should be well above 50% if the policy is using queue state")
else:
    print("No completed switches logged this episode.")

# correlation between |imbalance| and voting to switch, across all steps
imbs_all = np.array([i for i, a in action_log])
acts_all = np.array([a for i, a in action_log])
if len(set(acts_all)) > 1:
    corr = np.corrcoef(np.abs(imbs_all), acts_all)[0, 1]
    print(f"\nCorrelation(|imbalance|, switch-vote) across all steps: {corr:.3f}")
    print("  -> near 0 means the policy isn't voting based on imbalance magnitude")
else:
    print("\nPolicy only ever chose one action this episode -- can't compute correlation.")