"""
train_dqn.py

DQN counterpart to train.py. See train_dqn.py's original docstring
history for the fairness notes (reward config, MIN_GREEN, seeds,
timesteps, eval protocol all matched to train.py; buffer_size,
exploration schedule shape, target_update_interval left as
DQN-internal knobs with no PPO equivalent -- UNCHANGED here).

CHANGES IN THIS VERSION (2026-07-23, round 2):

  1. gradient_steps=4, learning_starts=10_000 -- unchanged from the
     previous round (see prior comments): targets the hypothesis that
     the Q-network wasn't getting enough updates to differentiate
     "switch" from "hold" by state.

  2. NEW: DirectionAgreementMonitorCallback. The corrected diagnostic
     run showed DQN's directional agreement at 36.3%/38.7% -- BELOW
     the 50% coin-flip line, not just "no pattern learned" (~50%).
     Neither change above specifically targets directionality, and
     guessing another hyperparameter blind isn't a substitute for
     seeing what's actually happening during training. This callback
     runs a short deterministic eval every `check_freq` steps using
     the SAME corrected agreement formula validated against
     test_diagnostic_multi.py / train.py's docstring:

         new_phase = 1 - phase   # phase is the OLD side; info is read
                                  # at the transition-to-yellow step,
                                  # before self._phase[tl] flips
         new_side_is_ns = (new_phase == 0)
         agrees = (mean_signed_imb > 0) == new_side_is_ns

     and logs mean hold duration + directional agreement per junction
     to TensorBoard (`custom/dir_agree_J1`, `custom/dir_agree_J2`,
     `custom/mean_hold_J1`, `custom/mean_hold_J2`) plus stdout, so you
     can watch the trend across training instead of waiting for the
     final checkpoint. If agreement is still below or near 50% by the
     later checkpoints, that's a signal the issue isn't "needs more
     training" and is worth stopping early to investigate rather than
     letting the full 500k steps run out on a policy that isn't
     improving on the metric that matters here.

  3. QValueSpreadCallback kept from the previous round -- still useful
     as a second, independent signal (whether the network differentiates
     actions AT ALL, separate from whether it differentiates them
     CORRECTLY).

Honesty note: none of this guarantees agreement ends up above 50%.
It gives you visibility into whether it's trending there during
training, which is what actually lets you decide whether to keep
this run, stop it early, or that the problem is somewhere other than
these DQN-internal knobs (e.g. worth checking whether the flattened
Discrete(4) action encoding in wrappers.py maps NS/EW switch intent
consistently for both junctions -- I don't have that file, so I can't
rule it in or out, but a below-chance result rather than a near-chance
result is the kind of pattern a systematic encoding issue would produce).
"""

import os
import sys
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.utils import get_linear_fn
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))

from env import SumoTrafficEnv2J
from wrappers import FlattenMultiDiscreteAction

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_FOLDER_NAME = "mg12_sw0.3_wd0.2"
MODELS_DIR = os.path.join(SCRIPT_DIR, "models_dqn", OUTPUT_FOLDER_NAME)
LOG_DIR = os.path.join(SCRIPT_DIR, "tb_logs_dqn", OUTPUT_FOLDER_NAME)

lr_schedule = get_linear_fn(start=3e-4, end=5e-5, end_fraction=1.0)  # same schedule as PPO

MAX_QUEUE = None
TOTAL_TIMESTEPS = 500_000  # matched to train.py

# --- reward + dynamics config -- MATCHED to train.py's actual constants.
#     UNCHANGED in this version. ---
SWITCH_PENALTY = 0.3
WASTED_VOTE_PENALTY = 0.03
IMBALANCE_BONUS_WEIGHT = 0.0
WRONG_DIRECTION_PENALTY = 0.2
MIN_GREEN = 12

TRAIN_PORT = 8823
EVAL_PORT = 8824
MONITOR_PORT = 8825  # separate port so the direction-agreement monitor
                      # doesn't collide with SB3's own EvalCallback env

TL_IDS = ["J1", "J2"]


def make_train_env(seed, port):
    def _init():
        env = SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            seed=seed,
            port=port,
            max_queue=MAX_QUEUE,
            switch_penalty=SWITCH_PENALTY,
            wasted_vote_penalty=WASTED_VOTE_PENALTY,
            imbalance_bonus_weight=IMBALANCE_BONUS_WEIGHT,
            wrong_direction_penalty=WRONG_DIRECTION_PENALTY,
            min_green=MIN_GREEN,
        )
        return FlattenMultiDiscreteAction(env)
    return _init


def make_eval_env(seed, port):
    def _init():
        env = SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            seed=seed,
            port=port,
            max_queue=MAX_QUEUE,
            switch_penalty=SWITCH_PENALTY,
            wasted_vote_penalty=WASTED_VOTE_PENALTY,
            imbalance_bonus_weight=IMBALANCE_BONUS_WEIGHT,
            wrong_direction_penalty=WRONG_DIRECTION_PENALTY,
            min_green=MIN_GREEN,
        )
        return FlattenMultiDiscreteAction(env)
    return _init


class QValueSpreadCallback(BaseCallback):
    """Diagnostic only. See previous version's docstring for rationale."""

    def __init__(self, check_freq: int = 5_000, batch_size: int = 256, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.batch_size = batch_size

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        buffer = self.model.replay_buffer
        if buffer.size() < self.batch_size:
            return True
        replay_data = buffer.sample(self.batch_size, env=self.model._vec_normalize_env)
        with torch.no_grad():
            q_values = self.model.q_net(replay_data.observations)
        spread = (q_values.max(dim=1).values - q_values.min(dim=1).values).mean().item()
        self.logger.record("custom/q_value_spread", spread)
        if self.verbose:
            print(f"[QValueSpread] step={self.num_timesteps} mean_q_spread={spread:.4f}")
        return True


class DirectionAgreementMonitorCallback(BaseCallback):
    """
    Runs a short deterministic eval every `check_freq` steps against a
    dedicated env (own port, separate from train/eval envs) and logs
    mean hold duration + directional agreement per junction, using the
    CORRECTED formula validated against train.py's docstring numbers
    (59.1%/57.6% for PPO) and test_diagnostic_multi.py post-fix.

    Not a substitute for the full test_diagnostic_multi.py run at the
    end (that uses 5 episodes; this uses n_episodes for speed) -- this
    is for watching the trend mid-training, not for final reporting
    numbers.
    """

    def __init__(self, port, check_freq: int = 50_000, n_episodes: int = 2,
                 max_steps: int = 3600, verbose: int = 1):
        super().__init__(verbose)
        self.port = port
        self.check_freq = check_freq
        self.n_episodes = n_episodes
        self.max_steps = max_steps

    def _make_env(self, seed):
        env = SumoTrafficEnv2J(
            cfg_path=os.path.join(SCRIPT_DIR, "network.sumocfg"),
            use_gui=False,
            max_steps=self.max_steps,
            seed=seed,
            port=self.port,
            switch_penalty=SWITCH_PENALTY,
            wasted_vote_penalty=WASTED_VOTE_PENALTY,
            imbalance_bonus_weight=IMBALANCE_BONUS_WEIGHT,
            wrong_direction_penalty=WRONG_DIRECTION_PENALTY,
            min_green=MIN_GREEN,
        )
        return FlattenMultiDiscreteAction(env)

    def _run_episode(self, seed):
        env = self._make_env(seed)
        obs, _ = env.reset()
        done = False

        log = {tl: {"hold_durations": [], "agrees": []} for tl in TL_IDS}
        current_hold_imbalance = {tl: [] for tl in TL_IDS}

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            for tl in TL_IDS:
                imb = info["imbalance"][tl]
                phase = info["phase"][tl]
                current_hold_imbalance[tl].append(imb)

                hold_dur = info["hold_duration_at_switch"][tl]
                if hold_dur is not None:
                    mean_signed_imb = float(np.mean(current_hold_imbalance[tl]))
                    log[tl]["hold_durations"].append(hold_dur)

                    # CORRECTED: phase read here is the OLD side (info
                    # is read at the transition-to-yellow step, before
                    # self._phase[tl] flips in env.py).
                    new_phase = 1 - phase
                    new_side_is_ns = (new_phase == 0)
                    agrees = (mean_signed_imb > 0) == new_side_is_ns
                    log[tl]["agrees"].append(agrees)

                    current_hold_imbalance[tl] = []

        env.close()
        return log

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        agg = {tl: {"hold_durations": [], "agrees": []} for tl in TL_IDS}
        for ep in range(self.n_episodes):
            log = self._run_episode(seed=ep)
            for tl in TL_IDS:
                agg[tl]["hold_durations"].extend(log[tl]["hold_durations"])
                agg[tl]["agrees"].extend(log[tl]["agrees"])

        print(f"\n[DirectionMonitor] step={self.num_timesteps}")
        for tl in TL_IDS:
            durations = agg[tl]["hold_durations"]
            agrees = agg[tl]["agrees"]
            mean_hold = float(np.mean(durations)) if durations else float("nan")
            agree_pct = 100 * float(np.mean(agrees)) if agrees else float("nan")
            self.logger.record(f"custom/mean_hold_{tl}", mean_hold)
            self.logger.record(f"custom/dir_agree_{tl}", agree_pct)
            flag = "" if agree_pct >= 50 else "  <-- below chance"
            print(f"  {tl}: mean_hold={mean_hold:.1f}  dir_agree={agree_pct:.1f}%{flag}")

        return True


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    best_dir = os.path.join(MODELS_DIR, "best")
    os.makedirs(best_dir, exist_ok=True)

    train_env = make_vec_env(make_train_env(42, TRAIN_PORT), n_envs=1)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=False)

    eval_env = make_vec_env(make_eval_env(0, EVAL_PORT), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    q_spread_callback = QValueSpreadCallback(check_freq=5_000, batch_size=256)
    direction_callback = DirectionAgreementMonitorCallback(
        port=MONITOR_PORT, check_freq=50_000, n_episodes=2,
    )

    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        policy_kwargs=dict(net_arch=[128, 128]),
        learning_rate=lr_schedule,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=128,
        train_freq=4,
        gradient_steps=4,
        target_update_interval=1_000,
        exploration_fraction=1.0,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        gamma=0.99,
        seed=42,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    print(f"\n{'='*70}")
    print(f"DQN training run: switch_penalty={SWITCH_PENALTY}, "
          f"wasted_vote_penalty={WASTED_VOTE_PENALTY}, "
          f"imbalance_bonus_weight={IMBALANCE_BONUS_WEIGHT}, "
          f"wrong_direction_penalty={WRONG_DIRECTION_PENALTY}, "
          f"min_green={MIN_GREEN}")
    print(f"gradient_steps=4, learning_starts=10_000")
    print(f"Monitoring dir. agreement every 50k steps against corrected formula.")
    print(f"Output -> {MODELS_DIR}")
    print(f"TensorBoard -> {LOG_DIR}")
    print(f"{'='*70}\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, q_spread_callback, direction_callback],
    )

    model.save(os.path.join(MODELS_DIR, "dqn_sumo_2junction"))
    train_env.save(os.path.join(MODELS_DIR, "vec_normalize_sumo_dqn.pkl"))

    train_env.close()
    eval_env.close()

    print(f"Training done -> {os.path.join(MODELS_DIR, 'dqn_sumo_2junction.zip')}")
    print(f"Rerun test_diagnostic_multi.py (corrected version) for the final report numbers.")


if __name__ == "__main__":
    main()