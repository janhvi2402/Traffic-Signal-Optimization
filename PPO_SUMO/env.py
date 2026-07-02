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

    Topology (from the net file):
          N1          N2
          |            |
    W ── J1 ────────  J2 ── E
          |            |
          S1          S2

    Traffic signal structure
    ─────────────────────────
    Both J1 and J2 use the same 4-phase programme (16 link indices each).
    The net file defines:
        Phase 0 (dur 42): GGggrrrrGGggrrrr  — N↔S green at both junctions
        Phase 1 (dur  3): yyyyrrrryyyyrrrr  — N↔S yellow
        Phase 2 (dur 42): rrrrGGggrrrrGGgg  — E↔W green at both junctions
        Phase 3 (dur  3): rrrryyyyrrrryyyy  — E↔W yellow

    Agent control
    ─────────────
    Action space : MultiDiscrete([2, 2])
        0 = keep current phase
        1 = request phase switch (triggers yellow → opposite green)

    NOTE: Phase timing is now FULLY agent/env controlled. SUMO's default
    tlLogic durations are overridden (setPhaseDuration=9999) immediately
    on every phase change, and both green (MIN_GREEN/MAX_GREEN) and
    yellow (YELLOW_TIME) durations are tracked and enforced manually in
    _apply_action. Nothing relies on SUMO's internal program timer.

    Observation (per junction, 7 features × 2 = 14 total)
        [0] mean queue length on NS incoming lanes   (normalised 0-1)
        [1] mean queue length on EW incoming lanes   (normalised 0-1)
        [2] mean waiting time on NS lanes            (normalised, cap 120 s)
        [3] mean waiting time on EW lanes            (normalised, cap 120 s)
        [4] current phase index (0=NS green, 1=EW green)
        [5] time spent in current phase              (normalised, cap 60 s)
        [6] 1 if currently in yellow, else 0

    Reward
    ──────
    Negative mean queue length across all incoming lanes of both junctions,
    normalised by max_queue.  Equivalent signal to the custom env.
    """

    #  SUMO IDs from network.net.xml 
    TL_IDS = ["J1", "J2"]

    # Incoming lanes per junction (order matches incLanes in net file)
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

    # Phase indices in the SUMO tlLogic programme
    PHASE_NS_GREEN  = 0
    PHASE_NS_YELLOW = 1
    PHASE_EW_GREEN  = 2
    PHASE_EW_YELLOW = 3

    #  constants 
    MAX_QUEUE   = 30    # vehicles per lane  (for normalisation)
    MAX_WAIT    = 120   # seconds            (for normalisation)
    MAX_PHASE_T = 60    # seconds            (for normalisation)
    MIN_GREEN   = 10    # minimum green duration before a switch is allowed
    MAX_GREEN   = 90    # seconds — hard cap so a phase can't run forever
    YELLOW_TIME = 3      # seconds — matches the net file's yellow duration

    def __init__(
        self,
        cfg_path    = None,
        use_gui     = False,
        max_steps   = 3600,
        seed        = None,
        port=8813
    ):
        super().__init__()

        if cfg_path is None:
            # default: same directory as this file
            cfg_path = os.path.join(os.path.dirname(__file__), "network.sumocfg")
        self.cfg_path  = os.path.abspath(cfg_path)
        self.use_gui   = use_gui
        self.max_steps = max_steps
        self._seed     = seed
        self.port = port
        # spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(14,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([2, 2])

        # internal state
        self._step_count      = 0
        self._phase           = {tl: 0 for tl in self.TL_IDS}   # 0=NS, 1=EW (logical)
        self._time_in_phase   = {tl: 0 for tl in self.TL_IDS}
        self._in_yellow       = {tl: False for tl in self.TL_IDS}
        self._time_in_yellow  = {tl: 0 for tl in self.TL_IDS}
        self._traci_started   = False

    # helpers

    def _start_sumo(self):
        binary = "sumo-gui" if self.use_gui else "sumo"
        cmd = [
            binary,
            "-c", self.cfg_path,
            "--no-step-log",
            "--no-warnings",
            "--random",
        ]
        if self._seed is not None:
            safe_seed = int(self._seed) % 2_147_483_647
            cmd += ["--seed", str(safe_seed)]

        # unique label per instance/port avoids "Connection 'default' is
        # already active" when train_env and eval_env run at the same time
        self.label = f"sim_{self.port}"
        traci.start(cmd, port=self.port, label=self.label)
        self.conn = traci.getConnection(self.label)
        self._traci_started = True

    def _close_sumo(self):
        if self._traci_started:
            self.conn.close()
            self._traci_started = False

    def _get_lane_stat(self, lane_id):
        """Return (queue_length, waiting_time) for a single lane."""
        q = self.conn.lane.getLastStepHaltingNumber(lane_id)
        w = self.conn.lane.getWaitingTime(lane_id)
        return q, w

    def _get_obs(self):
        obs = []
        for tl in self.TL_IDS:
            lanes = self.INCOMING_LANES[tl]

            # NS lanes
            ns_q, ns_w = zip(*[self._get_lane_stat(l) for l in lanes["NS"]])
            # EW lanes
            ew_q, ew_w = zip(*[self._get_lane_stat(l) for l in lanes["EW"]])

            obs.append(np.mean(ns_q) / self.MAX_QUEUE)
            obs.append(np.mean(ew_q) / self.MAX_QUEUE)
            obs.append(min(np.mean(ns_w) / self.MAX_WAIT, 1.0))
            obs.append(min(np.mean(ew_w) / self.MAX_WAIT, 1.0))
            obs.append(float(self._phase[tl]))                               # 0 or 1
            obs.append(min(self._time_in_phase[tl] / self.MAX_PHASE_T, 1.0))
            obs.append(float(self._in_yellow[tl]))

        return np.clip(np.array(obs, dtype=np.float32), 0.0, 1.0)

    def _get_reward(self):
        total_queue = 0.0
        n_lanes = 0
        for tl in self.TL_IDS:
            for group in self.INCOMING_LANES[tl].values():
                for lane in group:
                    total_queue += self.conn.lane.getLastStepHaltingNumber(lane)
                    n_lanes += 1
        return -(total_queue / n_lanes) / self.MAX_QUEUE

    def _apply_action(self, tl, action):
        if self._in_yellow[tl]:
            # manual yellow timer — SUMO's own program is frozen (duration=9999),
            # so we are the only thing advancing yellow -> green now.
            self._time_in_yellow[tl] += 1
            if self._time_in_yellow[tl] >= self.YELLOW_TIME:
                self._in_yellow[tl]      = False
                self._phase[tl]          = 1 - self._phase[tl]
                self._time_in_phase[tl]  = 0
                self._time_in_yellow[tl] = 0
                green_phase = self.PHASE_NS_GREEN if self._phase[tl] == 0 else self.PHASE_EW_GREEN
                self.conn.trafficlight.setPhase(tl, green_phase)
                self.conn.trafficlight.setPhaseDuration(tl, 9999)   # prevent auto-advance

        else:
            self._time_in_phase[tl] += 1
            force_switch = self._time_in_phase[tl] >= self.MAX_GREEN
            if (action == 1 and self._time_in_phase[tl] >= self.MIN_GREEN) or force_switch:
                yellow_phase = self.PHASE_NS_YELLOW if self._phase[tl] == 0 else self.PHASE_EW_YELLOW
                self.conn.trafficlight.setPhase(tl, yellow_phase)
                self.conn.trafficlight.setPhaseDuration(tl, 9999)   # prevent auto-advance
                self._in_yellow[tl]      = True
                self._time_in_yellow[tl] = 0

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

        self._start_sumo()

        for tl in self.TL_IDS:
            self.conn.trafficlight.setPhase(tl, self.PHASE_NS_GREEN)
            self.conn.trafficlight.setPhaseDuration(tl, 9999)   # prevent auto-advance from the very first step

        self.conn.simulationStep()

        return self._get_obs(), {}

    def step(self, action):
        assert len(action) == 2, "action must be [a_J1, a_J2]"

        for i, tl in enumerate(self.TL_IDS):
            self._apply_action(tl, int(action[i]))

        self.conn.simulationStep()
        self._step_count += 1

        obs     = self._get_obs()
        reward  = self._get_reward()
        done    = self._step_count >= self.max_steps

        return obs, reward, done, False, {}

    def close(self):
        self._close_sumo()

    def render(self, mode="human"):
        # GUI is enabled at construction time via use_gui=True;
        # this method is a no-op for headless runs.
        pass