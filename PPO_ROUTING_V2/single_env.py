import os
import sys
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from route_gen_single import generate_single_junction_routes

if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "SUMO_HOME not set. Add 'export SUMO_HOME=/path/to/sumo' to your shell profile."
    )
sys.path += [os.path.join(os.environ["SUMO_HOME"], "tools")]
import traci

from route_gen_single import generate_single_junction_routes


class SumoSingleJunctionEnv(gym.Env):
    """
    Single-junction Gymnasium environment.

    CHANGES vs the original version (fixing the shortcut-learning bug
    where the policy keyed off `time_in_phase` instead of queue state):

      1. Route file is regenerated with randomized, independently-sampled
         NS/EW demand on every reset() (see route_gen_single.py). This
         breaks the fixed mapping between elapsed phase time and queue
         buildup that let the policy ignore the queue features entirely.

      2. Observation now has 8 features (was 7): added an explicit
         NS-EW queue imbalance term at index [7]. This is redundant
         information (derivable from [0] and [1]) but puts the
         decision-relevant quantity directly in front of the network
         instead of requiring it to compute the difference itself.

      3. Optional feature dropout on `time_in_phase` (index [5]) during
         training: with probability `time_dropout_prob`, that feature is
         zeroed out for the step, forcing the policy to fall back on
         queue/wait features when the timing shortcut isn't available.
         Disabled at eval time (set time_dropout_prob=0.0).

    Observation (8 features):
        [0] mean queue length on NS incoming lanes   (normalised 0-1)
        [1] mean queue length on EW incoming lanes   (normalised 0-1)
        [2] mean waiting time on NS lanes            (normalised, cap 120s)
        [3] mean waiting time on EW lanes            (normalised, cap 120s)
        [4] current phase index (0=NS green, 1=EW green)
        [5] time spent in current phase               (normalised, cap 60s)
        [6] 1 if currently in yellow, else 0
        [7] NS-EW queue imbalance, signed             (normalised, [-1,1] -> stored [0,1])

    Action space: Discrete(2)
        0 = keep current phase
        1 = request phase switch (triggers yellow -> opposite green)

    Reward: negative mean queue length across incoming lanes, normalised
    by max_queue, PLUS a small bonus for switching when it actually
    resolves a real NS/EW imbalance (see _get_reward).
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

    def __init__(
        self,
        cfg_path=None,
        route_out_dir=None,
        use_gui=False,
        max_steps=3600,
        seed=None,
        port=8815,
        time_dropout_prob=0.0,   # set to 0.0 for eval/deployment
        randomize_routes=False,   # set to False to use a fixed route file
        
    ):
        super().__init__()

        if cfg_path is None:
            cfg_path = os.path.join(os.path.dirname(__file__), "network", "single_junction.sumocfg")
        self.cfg_path  = os.path.abspath(cfg_path)

        # where the generated route file goes; each port gets its own
        # file so parallel train/eval envs never clobber each other
        if route_out_dir is None:
            route_out_dir = os.path.join(os.path.dirname(__file__), "network", "_generated")
        self.route_out_path = os.path.join(route_out_dir, f"routes_port{port}.rou.xml")

        self.use_gui   = use_gui
        self.max_steps = max_steps
        self._seed     = seed
        self.port      = port
        self.time_dropout_prob = time_dropout_prob
        self.randomize_routes  = randomize_routes
        self._episode_count = 0

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space      = spaces.Discrete(2)

        self._step_count     = 0
        self._phase          = 0
        self._time_in_phase  = 0
        self._in_yellow      = False
        self._time_in_yellow = 0
        self._traci_started  = False
        self._just_switched  = False

    # helpers

    def _start_sumo(self):
        binary = "sumo-gui" if self.use_gui else "sumo"


        route_arg = []
        if self.randomize_routes:
            # unique seed per episode (not just per env) so every reset
            # gets genuinely different demand, not the same one repeated
            episode_seed = (self._seed or 0) * 100_003 + self._episode_count
            generate_single_junction_routes(
                self.route_out_path, episode_seed, sim_end=self.max_steps
            )
            # override the route file the .sumocfg points to
            route_arg = ["--route-files", self.route_out_path]

        cmd = [binary, "-c", self.cfg_path, "--no-step-log", "--no-warnings"] + route_arg
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

        ns_q_norm = np.mean(ns_q) / self.MAX_QUEUE
        ew_q_norm = np.mean(ew_q) / self.MAX_QUEUE

        # signed imbalance in [-1, 1], rescaled to [0, 1] for the Box space
        imbalance = (ns_q_norm - ew_q_norm)
        imbalance_scaled = (np.clip(imbalance, -1.0, 1.0) + 1.0) / 2.0

        time_in_phase_norm = min(self._time_in_phase / self.MAX_PHASE_T, 1.0)
        if self.time_dropout_prob > 0.0 and random.random() < self.time_dropout_prob:
            time_in_phase_norm = 0.0

        obs = [
            ns_q_norm,
            ew_q_norm,
            min(np.mean(ns_w) / self.MAX_WAIT, 1.0),
            min(np.mean(ew_w) / self.MAX_WAIT, 1.0),
            float(self._phase),
            time_in_phase_norm,
            float(self._in_yellow),
            imbalance_scaled,
        ]
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
                self._just_switched = True
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
        self._episode_count += 1

        self._step_count     = 0
        self._phase          = 0
        self._time_in_phase  = 0
        self._in_yellow      = False
        self._time_in_yellow = 0
        self._just_switched  = False

        self._start_sumo()
        self.conn.trafficlight.setPhase(self.TL_ID, self.PHASE_NS_GREEN)
        self.conn.trafficlight.setPhaseDuration(self.TL_ID, 9999)
        self.conn.simulationStep()
        self.prev_total_queue = 0
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