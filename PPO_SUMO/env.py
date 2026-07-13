# env.py
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# locate SUMO 
if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "SUMO_HOME not set. Add 'export SUMO_HOME=/path/to/sumo' to your shell profile."
    )
sys.path += [os.path.join(os.environ["SUMO_HOME"], "tools")]
import traci


class SumoTrafficEnv2J(gym.Env):
    """
    Gymnasium environment wrapping a real SUMO simulation for the
    2-junction network defined in network.net.xml.

    (docstring unchanged from before — see original for topology/phase details)

    NEW: info dict now reports raw per-junction NS/EW queue lengths,
    signed imbalance (ns_queue - ew_queue), and the hold duration
    (steps in phase) at the moment a switch actually fires. This is
    what lets you check, post-hoc, whether the agent's switch timing
    is actually correlated with which side is busier — or whether
    it's just cycling at the MIN_GREEN ceiling regardless of state.

    NEW: optional imbalance_bonus_weight (default 0.0, backward
    compatible). When > 0, adds a small positive reward each step a
    junction's current green phase matches whichever side (NS/EW)
    currently has the larger raw queue, scaled by the magnitude of
    the imbalance. This directly rewards "hold green on the busier
    side" instead of hoping it emerges purely from queue minimization.
    """

    TL_IDS = ["J1", "J2"]

    INCOMING_LANES = {
        "J1": {
            "NS": ["N1_J1_0", "S1_J1_0"],
            "EW": ["J2_J1_0", "W_J1_0"],
        },
        "J2": {
            "NS": ["N2_J2_0", "S2_J2_0"],
            "EW": ["E_J2_0",  "J1_J2_0"],
        },
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

    # NEW: default off. Try 0.02-0.05 if diagnostics show the agent
    # isn't conditioning on imbalance at all.
    IMBALANCE_BONUS_WEIGHT = 0.0

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
        imbalance_bonus_weight = None,   # NEW
    ):
        super().__init__()

        if cfg_path is None:
            cfg_path = os.path.join(os.path.dirname(__file__), "network.sumocfg")
        self.cfg_path  = os.path.abspath(cfg_path)
        self.use_gui   = use_gui
        self.max_steps = max_steps
        self._seed     = seed
        self.port = port
        self.max_queue = max_queue if max_queue is not None else self.MAX_QUEUE_DEFAULT
        self.switch_penalty = switch_penalty if switch_penalty is not None else self.SWITCH_PENALTY
        self.wasted_vote_penalty = (
            wasted_vote_penalty if wasted_vote_penalty is not None else self.WASTED_VOTE_PENALTY
        )
        # NEW
        self.imbalance_bonus_weight = (
            imbalance_bonus_weight if imbalance_bonus_weight is not None
            else self.IMBALANCE_BONUS_WEIGHT
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(14,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([2, 2])

        self._step_count      = 0
        self._phase           = {tl: 0 for tl in self.TL_IDS}
        self._time_in_phase   = {tl: 0 for tl in self.TL_IDS}
        self._in_yellow       = {tl: False for tl in self.TL_IDS}
        self._time_in_yellow  = {tl: 0 for tl in self.TL_IDS}
        self._traci_started   = False

        # NEW: hold duration captured right before a switch resets time_in_phase.
        # Read by step() to populate info["hold_duration_at_switch"].
        self._last_hold_duration = {tl: None for tl in self.TL_IDS}

    # helpers

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

    # NEW: raw (not normalised) NS/EW queue for one junction — used both
    # for observations-adjacent diagnostics and the imbalance bonus.
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

            obs.append(np.mean(ns_q) / self.max_queue)
            obs.append(np.mean(ew_q) / self.max_queue)
            obs.append(min(np.mean(ns_w) / self.MAX_WAIT, 1.0))
            obs.append(min(np.mean(ew_w) / self.MAX_WAIT, 1.0))
            obs.append(float(self._phase[tl]))
            obs.append(min(self._time_in_phase[tl] / self.MAX_PHASE_T, 1.0))
            obs.append(float(self._in_yellow[tl]))

        return np.clip(np.array(obs, dtype=np.float32), 0.0, 1.0)

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

        # NEW: imbalance bonus — reward holding green on the busier side.
        # Only applies while NOT in yellow (yellow has no "side" to reward).
        imbalance_term = 0.0
        if self.imbalance_bonus_weight > 0:
            for tl in self.TL_IDS:
                if self._in_yellow[tl]:
                    continue
                ns_q, ew_q = self._get_raw_ns_ew_queue(tl)
                imbalance = ns_q - ew_q          # positive => NS busier
                on_ns = (self._phase[tl] == 0)   # phase 0 = NS green
                # Reward sign: if NS busier (imbalance>0) and phase is NS -> positive.
                # If NS busier but phase is EW -> negative (wrong side).
                signed_alignment = imbalance if on_ns else -imbalance
                imbalance_term += self.imbalance_bonus_weight * (
                    signed_alignment / self.max_queue
                )

        return queue_term + switch_term + wasted_term + imbalance_term

    def _apply_action(self, tl, action):
        switched = False
        wasted_vote = False
        self._last_hold_duration[tl] = None  # NEW: reset each call

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
                # NEW: capture hold duration BEFORE it gets reset on yellow entry
                self._last_hold_duration[tl] = self._time_in_phase[tl]

                yellow_phase = self.PHASE_NS_YELLOW if self._phase[tl] == 0 else self.PHASE_EW_YELLOW
                self.conn.trafficlight.setPhase(tl, yellow_phase)
                self.conn.trafficlight.setPhaseDuration(tl, 9999)
                self._in_yellow[tl]      = True
                self._time_in_yellow[tl] = 0
                switched = (action == 1)

        return switched, wasted_vote

    # Gymnasium API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._close_sumo()

        if seed is not None:
            self._seed = seed

        self._step_count     = 0
        self._phase          = {tl: 0 for tl in self.TL_IDS}
        self._time_in_phase  = {tl: 0 for tl in self.TL_IDS}
        self._in_yellow      = {tl: False for tl in self.TL_IDS}
        self._time_in_yellow = {tl: 0 for tl in self.TL_IDS}
        self._last_hold_duration = {tl: None for tl in self.TL_IDS}  # NEW

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

        # NEW: raw queue + imbalance per junction, plus hold duration
        # at the moment of any switch that just fired this step.
        raw_queues = {tl: self._get_raw_ns_ew_queue(tl) for tl in self.TL_IDS}
        imbalance  = {tl: raw_queues[tl][0] - raw_queues[tl][1] for tl in self.TL_IDS}

        info = {
            "local_queue": {tl: self._get_local_queue(tl) for tl in self.TL_IDS},
            "switched": switches,
            "wasted_vote": wasted_votes,
            "ns_queue": {tl: raw_queues[tl][0] for tl in self.TL_IDS},   # NEW
            "ew_queue": {tl: raw_queues[tl][1] for tl in self.TL_IDS},   # NEW
            "imbalance": imbalance,                                      # NEW
            "phase": dict(self._phase),                                 # NEW
            "hold_duration_at_switch": dict(self._last_hold_duration),  # NEW
        }

        return obs, reward, done, False, info

    def close(self):
        self._close_sumo()

    def render(self, mode="human"):
        pass