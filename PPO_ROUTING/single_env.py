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


class SumoSingleJunctionEnv(gym.Env):
    """
    Single-junction Gymnasium environment. Deliberately uses the SAME
    7-feature observation layout and phase indexing as one junction's
    slice of SumoTrafficEnv2J (env.py), so a policy trained here can be
    applied directly, per-junction, to the 2-junction network at test
    time — see run_decentralized() in test.py for how that works.

    Observation (7 features):
        [0] mean queue length on NS incoming lanes   (normalised 0-1)
        [1] mean queue length on EW incoming lanes   (normalised 0-1)
        [2] mean waiting time on NS lanes            (normalised, cap 120s)
        [3] mean waiting time on EW lanes            (normalised, cap 120s)
        [4] current phase index (0=NS green, 1=EW green)
        [5] time spent in current phase               (normalised, cap 60s)
        [6] 1 if currently in yellow, else 0

    Action space: Discrete(2)
        0 = keep current phase
        1 = request phase switch (triggers yellow -> opposite green)

    Reward: negative mean queue length across incoming lanes, normalised
    by max_queue — identical formula to SumoTrafficEnv2J, so reward scale
    matches between single- and multi-junction settings.
    """

    TL_ID = "J"

    INCOMING_LANES = {
        "NS": ["N_J_0", "S_J_0"],
        "EW": ["E_J_0", "W_J_0"],
    }

    PHASE_NS_GREEN  = 0
    PHASE_NS_YELLOW = 1
    PHASE_EW_GREEN  = 2
    PHASE_EW_YELLOW = 3

    MAX_QUEUE   = 30
    MAX_WAIT    = 120
    MAX_PHASE_T = 60
    MIN_GREEN   = 10
    MAX_GREEN   = 90
    YELLOW_TIME = 3

    def __init__(self, cfg_path=None, use_gui=False, max_steps=3600, seed=None, port=8815):
        super().__init__()

        if cfg_path is None:
            cfg_path = os.path.join(os.path.dirname(__file__), "network", "single_junction.sumocfg")
        self.cfg_path  = os.path.abspath(cfg_path)
        self.use_gui   = use_gui
        self.max_steps = max_steps
        self._seed     = seed
        self.port      = port

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)
        self.action_space      = spaces.Discrete(2)

        self._step_count     = 0
        self._phase          = 0
        self._time_in_phase  = 0
        self._in_yellow      = False
        self._time_in_yellow = 0
        self._traci_started  = False

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

    def _get_obs(self):
        ns_q, ns_w = zip(*[self._get_lane_stat(l) for l in self.INCOMING_LANES["NS"]])
        ew_q, ew_w = zip(*[self._get_lane_stat(l) for l in self.INCOMING_LANES["EW"]])

        obs = [
            np.mean(ns_q) / self.MAX_QUEUE,
            np.mean(ew_q) / self.MAX_QUEUE,
            min(np.mean(ns_w) / self.MAX_WAIT, 1.0),
            min(np.mean(ew_w) / self.MAX_WAIT, 1.0),
            float(self._phase),
            min(self._time_in_phase / self.MAX_PHASE_T, 1.0),
            float(self._in_yellow),
        ]
        return np.clip(np.array(obs, dtype=np.float32), 0.0, 1.0)

    def _get_reward(self):
        total_queue = 0.0
        n_lanes = 0
        for group in self.INCOMING_LANES.values():
            for lane in group:
                total_queue += self.conn.lane.getLastStepHaltingNumber(lane)
                n_lanes += 1
        return -(total_queue / n_lanes) / self.MAX_QUEUE

    def _apply_action(self, action):
        if self._in_yellow:
            self._time_in_yellow += 1
            if self._time_in_yellow >= self.YELLOW_TIME:
                self._in_yellow      = False
                self._phase          = 1 - self._phase
                self._time_in_phase  = 0
                self._time_in_yellow = 0
                green_phase = self.PHASE_NS_GREEN if self._phase == 0 else self.PHASE_EW_GREEN
                self.conn.trafficlight.setPhase(self.TL_ID, green_phase)
                self.conn.trafficlight.setPhaseDuration(self.TL_ID, 9999)
        else:
            self._time_in_phase += 1
            force_switch = self._time_in_phase >= self.MAX_GREEN
            if (action == 1 and self._time_in_phase >= self.MIN_GREEN) or force_switch:
                yellow_phase = self.PHASE_NS_YELLOW if self._phase == 0 else self.PHASE_EW_YELLOW
                self.conn.trafficlight.setPhase(self.TL_ID, yellow_phase)
                self.conn.trafficlight.setPhaseDuration(self.TL_ID, 9999)
                self._in_yellow      = True
                self._time_in_yellow = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._close_sumo()

        if seed is not None:
            self._seed = seed

        self._step_count     = 0
        self._phase          = 0
        self._time_in_phase  = 0
        self._in_yellow      = False
        self._time_in_yellow = 0

        self._start_sumo()
        self.conn.trafficlight.setPhase(self.TL_ID, self.PHASE_NS_GREEN)
        self.conn.trafficlight.setPhaseDuration(self.TL_ID, 9999)
        self.conn.simulationStep()

        return self._get_obs(), {}

    def step(self, action):
        self._apply_action(int(action))
        self.conn.simulationStep()
        self._step_count += 1

        obs    = self._get_obs()
        reward = self._get_reward()
        done   = self._step_count >= self.max_steps

        return obs, reward, done, False, {}

    def close(self):
        self._close_sumo()

    def render(self, mode="human"):
        pass
