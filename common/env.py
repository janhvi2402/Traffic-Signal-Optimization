# env.py
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "SUMO_HOME not set. Add 'export SUMO_HOME=/path/to/sumo' to your shell profile."
    )
sys.path += [os.path.join(os.environ["SUMO_HOME"], "tools")]
import traci


class SumoTrafficEnv2J(gym.Env):
    """
    (docstring: topology/phase details unchanged from original)

    Seed rotation, imbalance_bonus_weight, wrong_direction_penalty: all
    unchanged from the previous version — see prior notes.

    NEW: observation now includes an explicit signed imbalance feature
    per junction (ns_queue - ew_queue, normalised to [-1, 1]), on top of
    the existing raw ns_queue/ew_queue values. Rationale: the network
    previously had to implicitly learn to subtract two noisy raw
    values to detect imbalance. Handing it the difference directly
    removes one layer of required inference — cheap to add, doesn't
    change reward or action logic at all. Observation shape: 14 -> 16
    (8 features x 2 junctions instead of 7 x 2). This means a model
    trained on the new observation space CANNOT load weights from a
    model trained on the old 14-dim space — this is a fresh-training
    change, not a fine-tune.
    """

    TL_IDS = ["J1", "J2"]

    INCOMING_LANES = {
        "J1": {"NS": ["N1_J1_0", "S1_J1_0"], "EW": ["J2_J1_0", "W_J1_0"]},
        "J2": {"NS": ["N2_J2_0", "S2_J2_0"], "EW": ["E_J2_0",  "J1_J2_0"]},
    }

    PHASE_NS_GREEN  = 0
    PHASE_NS_YELLOW = 1
    PHASE_EW_GREEN  = 2
    PHASE_EW_YELLOW = 3

    MAX_QUEUE_DEFAULT = 30
    MAX_WAIT    = 120
    MAX_PHASE_T = 60
    MIN_GREEN   = 10
    MAX_GREEN   = 90
    YELLOW_TIME = 3

    SWITCH_PENALTY = 0.15
    WASTED_VOTE_PENALTY = 0.03
    IMBALANCE_BONUS_WEIGHT = 0.0
    WRONG_DIRECTION_PENALTY = 0.0

    def __init__(
        self,
        cfg_path    = None,
        use_gui     = False,
        max_steps   = 3600,
        seed        = None,
        port=8813,
        max_queue   = None,
        switch_penalty = None,
        wasted_vote_penalty = None,
        imbalance_bonus_weight = None,
        wrong_direction_penalty = None,
    ):
        super().__init__()

        if cfg_path is None:
            cfg_path = os.path.join(os.path.dirname(__file__), "network.sumocfg")
        self.cfg_path  = os.path.abspath(cfg_path)
        self.use_gui   = use_gui
        self.max_steps = max_steps
        self.port = port
        self.max_queue = max_queue if max_queue is not None else self.MAX_QUEUE_DEFAULT
        self.switch_penalty = switch_penalty if switch_penalty is not None else self.SWITCH_PENALTY
        self.wasted_vote_penalty = (
            wasted_vote_penalty if wasted_vote_penalty is not None else self.WASTED_VOTE_PENALTY
        )
        self.imbalance_bonus_weight = (
            imbalance_bonus_weight if imbalance_bonus_weight is not None
            else self.IMBALANCE_BONUS_WEIGHT
        )
        self.wrong_direction_penalty = (
            wrong_direction_penalty if wrong_direction_penalty is not None
            else self.WRONG_DIRECTION_PENALTY
        )

        self._base_seed = seed
        self._auto_seed_rng = np.random.default_rng(seed)
        self._seed = seed
        self._episode_count = 0

        # NEW: shape 14 -> 16, bounds -1..1 to allow the signed imbalance
        # feature. All other 14 features are still in [0, 1]; np.clip in
        # _get_obs() keeps them there, only the imbalance slot uses the
        # wider range.
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(16,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([2, 2])

        self._step_count      = 0
        self._phase           = {tl: 0 for tl in self.TL_IDS}
        self._time_in_phase   = {tl: 0 for tl in self.TL_IDS}
        self._in_yellow       = {tl: False for tl in self.TL_IDS}
        self._time_in_yellow  = {tl: 0 for tl in self.TL_IDS}
        self._traci_started   = False
        self._last_hold_duration = {tl: None for tl in self.TL_IDS}
        self._wrong_direction_this_step = {tl: False for tl in self.TL_IDS}

    def _start_sumo(self):
        binary = "sumo-gui" if self.use_gui else "sumo"
        cmd = [binary, "-c", self.cfg_path, "--no-step-log", "--no-warnings"]
        if self._seed is not None:
            safe_seed = int(self._seed) % 2_147_483_647
            cmd += ["--seed", str(safe_seed)]
        else:
            cmd += ["--random"]
        self.label = f"sim_{self.port}"
        traci.start(cmd, port=self.port, label=self.label)
        self.conn = traci.getConnection(self.label)
        self._traci_started = True

    def _close_sumo(self):
        if self._traci_started:
            self.conn.close()
            self._traci_started = False

    def _get_lane_stat(self, lane_id):
        q = self.conn.lane.getLastStepHaltingNumber(lane_id)
        w = self.conn.lane.getWaitingTime(lane_id)
        return q, w

    def _get_raw_ns_ew_queue(self, tl):
        lanes = self.INCOMING_LANES[tl]
        ns_q = np.mean([self.conn.lane.getLastStepHaltingNumber(l) for l in lanes["NS"]])
        ew_q = np.mean([self.conn.lane.getLastStepHaltingNumber(l) for l in lanes["EW"]])
        return float(ns_q), float(ew_q)

    def _get_obs(self):
        obs = []
        for tl in self.TL_IDS:
            lanes = self.INCOMING_LANES[tl]
            ns_q, ns_w = zip(*[self._get_lane_stat(l) for l in lanes["NS"]])
            ew_q, ew_w = zip(*[self._get_lane_stat(l) for l in lanes["EW"]])

            ns_q_mean = float(np.mean(ns_q))
            ew_q_mean = float(np.mean(ew_q))

            obs.append(np.clip(ns_q_mean / self.max_queue, 0.0, 1.0))
            obs.append(np.clip(ew_q_mean / self.max_queue, 0.0, 1.0))
            obs.append(min(np.mean(ns_w) / self.MAX_WAIT, 1.0))
            obs.append(min(np.mean(ew_w) / self.MAX_WAIT, 1.0))
            obs.append(float(self._phase[tl]))
            obs.append(min(self._time_in_phase[tl] / self.MAX_PHASE_T, 1.0))
            obs.append(float(self._in_yellow[tl]))

            # NEW: explicit signed imbalance feature, normalised to [-1, 1]
            imbalance_norm = np.clip(
                (ns_q_mean - ew_q_mean) / self.max_queue, -1.0, 1.0
            )
            obs.append(imbalance_norm)

        return np.array(obs, dtype=np.float32)

    def _get_local_queue(self, tl):
        total, n = 0.0, 0
        for group in self.INCOMING_LANES[tl].values():
            for lane in group:
                total += self.conn.lane.getLastStepHaltingNumber(lane)
                n += 1
        return total / n

    def _get_reward(self, switches_this_step, wasted_votes_this_step):
        total_queue = 0.0
        n_lanes = 0
        for tl in self.TL_IDS:
            for group in self.INCOMING_LANES[tl].values():
                for lane in group:
                    total_queue += self.conn.lane.getLastStepHaltingNumber(lane)
                    n_lanes += 1
        queue_term  = -(total_queue / n_lanes) / self.max_queue
        switch_term = -self.switch_penalty * switches_this_step
        wasted_term = -self.wasted_vote_penalty * wasted_votes_this_step

        imbalance_term = 0.0
        if self.imbalance_bonus_weight > 0:
            for tl in self.TL_IDS:
                if self._in_yellow[tl]:
                    continue
                ns_q, ew_q = self._get_raw_ns_ew_queue(tl)
                imbalance = ns_q - ew_q
                on_ns = (self._phase[tl] == 0)
                signed_alignment = imbalance if on_ns else -imbalance
                imbalance_term += self.imbalance_bonus_weight * (signed_alignment / self.max_queue)

        wrong_dir_term = -self.wrong_direction_penalty * sum(self._wrong_direction_this_step.values())

        return queue_term + switch_term + wasted_term + imbalance_term + wrong_dir_term

    def _apply_action(self, tl, action):
        switched = False
        wasted_vote = False
        self._last_hold_duration[tl] = None
        self._wrong_direction_this_step[tl] = False

        if self._in_yellow[tl]:
            if action == 1:
                wasted_vote = True
            self._time_in_yellow[tl] += 1
            if self._time_in_yellow[tl] >= self.YELLOW_TIME:
                self._in_yellow[tl]      = False
                self._phase[tl]          = 1 - self._phase[tl]
                self._time_in_phase[tl]  = 0
                self._time_in_yellow[tl] = 0
                green_phase = self.PHASE_NS_GREEN if self._phase[tl] == 0 else self.PHASE_EW_GREEN
                self.conn.trafficlight.setPhase(tl, green_phase)
                self.conn.trafficlight.setPhaseDuration(tl, 9999)

        else:
            self._time_in_phase[tl] += 1
            force_switch = self._time_in_phase[tl] >= self.MAX_GREEN
            eligible = self._time_in_phase[tl] >= self.MIN_GREEN

            if action == 1 and not eligible and not force_switch:
                wasted_vote = True

            if (action == 1 and eligible) or force_switch:
                wrong_direction = False
                if action == 1 and not force_switch:
                    ns_q, ew_q = self._get_raw_ns_ew_queue(tl)
                    currently_on_ns = (self._phase[tl] == 0)
                    leaving_busier_side = (currently_on_ns and ns_q > ew_q) or \
                                          (not currently_on_ns and ew_q > ns_q)
                    wrong_direction = leaving_busier_side

                self._last_hold_duration[tl] = self._time_in_phase[tl]
                yellow_phase = self.PHASE_NS_YELLOW if self._phase[tl] == 0 else self.PHASE_EW_YELLOW
                self.conn.trafficlight.setPhase(tl, yellow_phase)
                self.conn.trafficlight.setPhaseDuration(tl, 9999)
                self._in_yellow[tl]      = True
                self._time_in_yellow[tl] = 0
                switched = (action == 1)
                self._wrong_direction_this_step[tl] = wrong_direction

        return switched, wasted_vote

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._close_sumo()

        if seed is not None:
            self._seed = seed
        elif self._episode_count == 0 and self._base_seed is not None:
            self._seed = self._base_seed
        else:
            self._seed = int(self._auto_seed_rng.integers(0, 2_147_483_647))

        self._episode_count += 1

        self._step_count     = 0
        self._phase          = {tl: 0 for tl in self.TL_IDS}
        self._time_in_phase  = {tl: 0 for tl in self.TL_IDS}
        self._in_yellow      = {tl: False for tl in self.TL_IDS}
        self._time_in_yellow = {tl: 0 for tl in self.TL_IDS}
        self._last_hold_duration = {tl: None for tl in self.TL_IDS}
        self._wrong_direction_this_step = {tl: False for tl in self.TL_IDS}

        self._start_sumo()

        for tl in self.TL_IDS:
            self.conn.trafficlight.setPhase(tl, self.PHASE_NS_GREEN)
            self.conn.trafficlight.setPhaseDuration(tl, 9999)

        self.conn.simulationStep()

        return self._get_obs(), {}

    def step(self, action):
        assert len(action) == 2, "action must be [a_J1, a_J2]"

        switches = {}
        wasted_votes = {}
        for i, tl in enumerate(self.TL_IDS):
            switches[tl], wasted_votes[tl] = self._apply_action(tl, int(action[i]))

        self.conn.simulationStep()
        self._step_count += 1

        n_switches = sum(switches.values())
        n_wasted   = sum(wasted_votes.values())

        obs     = self._get_obs()
        reward  = self._get_reward(n_switches, n_wasted)
        done    = self._step_count >= self.max_steps

        raw_queues = {tl: self._get_raw_ns_ew_queue(tl) for tl in self.TL_IDS}
        imbalance  = {tl: raw_queues[tl][0] - raw_queues[tl][1] for tl in self.TL_IDS}

        info = {
            "local_queue": {tl: self._get_local_queue(tl) for tl in self.TL_IDS},
            "switched": switches,
            "wasted_vote": wasted_votes,
            "ns_queue": {tl: raw_queues[tl][0] for tl in self.TL_IDS},
            "ew_queue": {tl: raw_queues[tl][1] for tl in self.TL_IDS},
            "imbalance": imbalance,
            "phase": dict(self._phase),
            "hold_duration_at_switch": dict(self._last_hold_duration),
            "wrong_direction": dict(self._wrong_direction_this_step),
            "sumo_seed": self._seed,
        }

        return obs, reward, done, False, info

    def close(self):
        self._close_sumo()

    def render(self, mode="human"):
        pass