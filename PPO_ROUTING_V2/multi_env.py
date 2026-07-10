import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from route_gen_multi import generate_multi_junction_routes

if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "SUMO_HOME not set. Add 'export SUMO_HOME=/path/to/sumo' to your shell profile."
    )
sys.path += [os.path.join(os.environ["SUMO_HOME"], "tools")]
import traci


class SumoMultiJunctionEnv(gym.Env):
    """
    2-junction Gymnasium environment.

    CHANGED: observation is now 8 features per junction (16 total, was
    14) to match the updated SumoSingleJunctionEnv — added an explicit
    NS-EW queue imbalance term at index [7] of each junction's slice.
    This keeps a single-junction-trained policy's input layout valid
    when applied decentralized to J1 and J2 here (see run_decentralized).

    Observation (per junction, 8 features x 2 = 16 total)
        [0] mean queue length on NS incoming lanes   (normalised 0-1)
        [1] mean queue length on EW incoming lanes   (normalised 0-1)
        [2] mean waiting time on NS lanes            (normalised, cap 120 s)
        [3] mean waiting time on EW lanes            (normalised, cap 120 s)
        [4] current phase index (0=NS green, 1=EW green)
        [5] time spent in current phase              (normalised, cap 60 s)
        [6] 1 if currently in yellow, else 0
        [7] NS-EW queue imbalance, signed             (normalised, [0,1])
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

    MAX_QUEUE   = 30
    MAX_WAIT    = 120
    MAX_PHASE_T = 60
    MIN_GREEN   = 10
    MAX_GREEN   = 90
    YELLOW_TIME = 3

    def __init__(
    self,
    cfg_path=None,
    route_out_dir=None,
    use_gui=False,
    max_steps=3600,
    seed=None,
    port=8813,
    randomize_routes=False,
    ):
        super().__init__()

        if cfg_path is None:
            cfg_path = os.path.join(os.path.dirname(__file__), "network", "multi_junction.sumocfg")
        self.cfg_path  = os.path.abspath(cfg_path)
        if route_out_dir is None:
            route_out_dir = os.path.join(
                os.path.dirname(__file__),
                "network",
                "_generated"
            )

        self.route_out_path = os.path.join(
            route_out_dir,
            f"multi_routes_{port}.rou.xml"
        )

        self.randomize_routes = randomize_routes
        self._episode_count = 0
        self.use_gui   = use_gui
        self.max_steps = max_steps
        self._seed     = seed
        self.port = port

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([2, 2])

        self._step_count      = 0
        self._phase           = {tl: 0 for tl in self.TL_IDS}
        self._time_in_phase   = {tl: 0 for tl in self.TL_IDS}
        self._in_yellow       = {tl: False for tl in self.TL_IDS}
        self._time_in_yellow  = {tl: 0 for tl in self.TL_IDS}
        self._just_switched   = {tl: False for tl in self.TL_IDS}
        self._traci_started   = False

    def _start_sumo(self):
        binary = "sumo-gui" if self.use_gui else "sumo"

        route_arg = []

        if self.randomize_routes:

            episode_seed = (self._seed or 0) * 100003 + self._episode_count

            generate_multi_junction_routes(
                self.route_out_path,
                episode_seed,
                sim_end=self.max_steps
            )

            route_arg = [
                "--route-files",
                self.route_out_path
            ]

        cmd = [
            binary,
            "-c",
            self.cfg_path,
            "--no-step-log",
            "--no-warnings"
        ] + route_arg
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
        obs = []
        for tl in self.TL_IDS:
            lanes = self.INCOMING_LANES[tl]

            ns_q, ns_w = zip(*[self._get_lane_stat(l) for l in lanes["NS"]])
            ew_q, ew_w = zip(*[self._get_lane_stat(l) for l in lanes["EW"]])

            ns_q_norm = np.mean(ns_q) / self.MAX_QUEUE
            ew_q_norm = np.mean(ew_q) / self.MAX_QUEUE
            imbalance = np.clip(ns_q_norm - ew_q_norm, -1.0, 1.0)
            imbalance_scaled = (imbalance + 1.0) / 2.0

            obs.append(ns_q_norm)
            obs.append(ew_q_norm)
            obs.append(min(np.mean(ns_w) / self.MAX_WAIT, 1.0))
            obs.append(min(np.mean(ew_w) / self.MAX_WAIT, 1.0))
            obs.append(float(self._phase[tl]))
            obs.append(min(self._time_in_phase[tl] / self.MAX_PHASE_T, 1.0))
            obs.append(float(self._in_yellow[tl]))
            obs.append(imbalance_scaled)

        return np.clip(np.array(obs, dtype=np.float32), 0.0, 1.0)

    def _get_reward(self):
        total_queue = 0.0
        total_wait = 0.0
        n_lanes = 0

        ns_q_total = 0.0
        ew_q_total = 0.0
        ns_n = 0
        ew_n = 0

        for axis, group in self.INCOMING_LANES.items():
            for lane in group:

                q = self.conn.lane.getLastStepHaltingNumber(lane)
                w = self.conn.lane.getWaitingTime(lane)

                total_queue += q
                total_wait += w
                n_lanes += 1

                if axis == "NS":
                    ns_q_total += q
                    ns_n += 1
                else:
                    ew_q_total += q
                    ew_n += 1

        # Average queue penalty
        queue_penalty = -(total_queue / n_lanes) / self.MAX_QUEUE

        # Average waiting-time penalty
        wait_penalty = -(total_wait / n_lanes) / self.MAX_WAIT

        # Weight queue more than waiting time
        reward = (
            0.7 * queue_penalty +
            0.3 * wait_penalty
        )

        # Bonus only when switching helps an imbalanced intersection
        if self._just_switched:
            imbalance = abs(
                (ns_q_total / ns_n) -
                (ew_q_total / ew_n)
            ) / self.MAX_QUEUE

            reward += 0.05 * imbalance

        self._just_switched = False

        return reward

    def _apply_action(self, tl, action):
        if self._in_yellow[tl]:
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
            if (action == 1 and self._time_in_phase[tl] >= self.MIN_GREEN) or force_switch:
                yellow_phase = self.PHASE_NS_YELLOW if self._phase[tl] == 0 else self.PHASE_EW_YELLOW
                self.conn.trafficlight.setPhase(tl, yellow_phase)
                self.conn.trafficlight.setPhaseDuration(tl, 9999)
                self._in_yellow[tl]      = True
                self._time_in_yellow[tl] = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._close_sumo()

        if seed is not None:
            self._seed = seed
            self._episode_count += 1

        self._step_count     = 0
        self._phase          = {tl: 0 for tl in self.TL_IDS}
        self._time_in_phase  = {tl: 0 for tl in self.TL_IDS}
        self._in_yellow      = {tl: False for tl in self.TL_IDS}
        self._time_in_yellow = {tl: 0 for tl in self.TL_IDS}

        self._start_sumo()

        for tl in self.TL_IDS:
            self.conn.trafficlight.setPhase(tl, self.PHASE_NS_GREEN)
            self.conn.trafficlight.setPhaseDuration(tl, 9999)

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
        pass