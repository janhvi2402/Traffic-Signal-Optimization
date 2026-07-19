import os
import sys
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


class SumoSingleJunctionEnv(gym.Env):
    """
    TRAIN on this env. Test the resulting model on SumoMultiJunctionEnv
    (multi_env.py) by splitting the 16-dim obs into two 8-dim halves.

    This brings single_env.py in line 1:1 with the centralized 2-junction
    env.py's proven fixes, adapted to one junction:

      - MIN_GREEN raised 10 -> 15 (FIXED, not randomized). Same reasoning
        as centralized: diagnostics showed switches clustering exactly at
        the old floor: raising it removes that specific escape hatch
        outright, rather than trying to average it out with a random
        range.
      - SWITCH_PENALTY 0.15 -> 0.4, WRONG_DIRECTION_PENALTY (new) = 0.25:
        same values that measurably improved the centralized run. The old
        SWITCH_BONUS_WEIGHT (a bonus for switching toward the heavier
        side) is REMOVED -- diagnosed as a "+reward for switching whenever
        any imbalance exists" shortcut, since some imbalance is nearly
        always present under randomized demand. WRONG_DIRECTION_PENALTY
        replaces it: a flat, symmetric cost for abandoning the side that
        was still busier, computed the same way centralized does it (a
        snapshot at the moment a voluntary switch is requested, compared
        against which side is actually heavier at that moment).
      - Seed rotation is IDENTICAL to centralized's: `seed` seeds a
        rotation RNG; the first reset() uses it literally, every
        subsequent reset() (the normal case during SB3 training) draws a
        new SUMO seed.
      - info dict field names now MATCH centralized's exactly
        (local_queue, switched, wasted_vote, ns_queue, ew_queue,
        imbalance, phase, hold_duration_at_switch, wrong_direction,
        sumo_seed) so diagnostic/plotting code is portable between the
        two projects with minimal changes.

    Deliberately KEPT DIFFERENT from centralized (both are necessary for
    the single-junction-train / multi-junction-test workflow, not bugs):
      - randomize_routes=True by default: centralized trains directly on
        the real 2-junction network and never needed synthetic demand.
        single_env.py trains on an isolated single junction, so route_gen
        supplies the domain-randomized demand that stands in for "the
        rest of the network's traffic arriving here".
      - Observation is 8 features (not 7): includes an explicit signed
        NS-EW imbalance feature at index [7]. multi_env.py's 16-dim obs
        is built by concatenating two of these 8-feature blocks, so this
        single-junction policy can be applied independently to J1 and J2
        without retraining -- that split only works if the two envs
        agree on this 8-feature layout.

    Observation (8 features):
        [0] mean queue length on NS incoming lanes   (normalised 0-1)
        [1] mean queue length on EW incoming lanes   (normalised 0-1)
        [2] mean waiting time on NS lanes            (normalised, cap 120s)
        [3] mean waiting time on EW lanes            (normalised, cap 120s)
        [4] current phase index (0=NS green, 1=EW green)
        [5] time spent in current phase               (normalised, cap 60s)
        [6] 1 if currently in yellow, else 0
        [7] NS-EW queue imbalance, signed             ([-1,1] -> stored [0,1])
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

    MAX_QUEUE_DEFAULT = 30
    MAX_WAIT    = 120
    MAX_PHASE_T = 60
    MIN_GREEN   = 15   # was 10, matching centralized's raised floor
    MAX_GREEN   = 90
    YELLOW_TIME = 3

    SWITCH_PENALTY           = 0.4    # was SWITCH_BONUS_WEIGHT=0.05/SWITCH_PENALTY=0.15
    WASTED_VOTE_PENALTY      = 0.03   # was 0.01
    IMBALANCE_BONUS_WEIGHT   = 0.0    # off by default, matching centralized's chosen config
    WRONG_DIRECTION_PENALTY  = 0.25   # new -- replaces the old switch bonus

    def __init__(
        self,
        cfg_path=None,
        route_out_dir=None,
        use_gui=False,
        max_steps=3600,
        seed=None,
        port=8815,
        randomize_routes=True,
        max_queue=None,
        switch_penalty=None,
        wasted_vote_penalty=None,
        imbalance_bonus_weight=None,
        wrong_direction_penalty=None,
    ):
        super().__init__()

        if cfg_path is None:
            cfg_path = os.path.join(os.path.dirname(__file__), "network", "single_junction.sumocfg")
        self.cfg_path = os.path.abspath(cfg_path)

        if route_out_dir is None:
            route_out_dir = os.path.join(os.path.dirname(__file__), "network", "_generated")
        self.route_out_path = os.path.join(route_out_dir, f"routes_port{port}.rou.xml")

        self.use_gui   = use_gui
        self.max_steps = max_steps
        self.port      = port
        self.randomize_routes = randomize_routes

        self.max_queue = max_queue if max_queue is not None else self.MAX_QUEUE_DEFAULT
        self.switch_penalty = switch_penalty if switch_penalty is not None else self.SWITCH_PENALTY
        self.wasted_vote_penalty = (
            wasted_vote_penalty if wasted_vote_penalty is not None else self.WASTED_VOTE_PENALTY
        )
        self.imbalance_bonus_weight = (
            imbalance_bonus_weight if imbalance_bonus_weight is not None else self.IMBALANCE_BONUS_WEIGHT
        )
        self.wrong_direction_penalty = (
            wrong_direction_penalty if wrong_direction_penalty is not None else self.WRONG_DIRECTION_PENALTY
        )

        # seed rotation -- identical scheme to centralized env.py.
        self._base_seed = seed
        self._auto_seed_rng = np.random.default_rng(seed)
        self._seed = seed
        self._episode_count = 0

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space      = spaces.Discrete(2)

        self._step_count     = 0
        self._phase          = 0
        self._time_in_phase  = 0
        self._in_yellow      = False
        self._time_in_yellow = 0
        self._traci_started  = False

        self._last_hold_duration = None
        self._wrong_direction_this_step = False

    # ---------------------------------------------------------------- #
    # sumo lifecycle
    # ---------------------------------------------------------------- #

    def _start_sumo(self):
        binary = "sumo-gui" if self.use_gui else "sumo"

        route_arg = []
        if self.randomize_routes:
            episode_seed = (self._seed or 0) * 100_003 + self._episode_count
            generate_single_junction_routes(
                self.route_out_path, episode_seed, sim_end=self.max_steps
            )
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

    # ---------------------------------------------------------------- #
    # observation / reward helpers
    # ---------------------------------------------------------------- #

    def _get_lane_stat(self, lane_id):
        q = self.conn.lane.getLastStepHaltingNumber(lane_id)
        w = self.conn.lane.getWaitingTime(lane_id)
        return q, w

    def _get_raw_ns_ew_queue(self):
        """Raw (unnormalized) mean halting vehicle count, NS and EW."""
        ns_q = np.mean([self.conn.lane.getLastStepHaltingNumber(l) for l in self.INCOMING_LANES["NS"]])
        ew_q = np.mean([self.conn.lane.getLastStepHaltingNumber(l) for l in self.INCOMING_LANES["EW"]])
        return float(ns_q), float(ew_q)

    def _get_local_queue(self):
        total, n = 0.0, 0
        for group in self.INCOMING_LANES.values():
            for lane in group:
                total += self.conn.lane.getLastStepHaltingNumber(lane)
                n += 1
        return total / n

    def _get_obs(self):
        ns_q, ns_w = zip(*[self._get_lane_stat(l) for l in self.INCOMING_LANES["NS"]])
        ew_q, ew_w = zip(*[self._get_lane_stat(l) for l in self.INCOMING_LANES["EW"]])

        ns_q_norm = np.mean(ns_q) / self.max_queue
        ew_q_norm = np.mean(ew_q) / self.max_queue

        imbalance = ns_q_norm - ew_q_norm
        imbalance_scaled = (np.clip(imbalance, -1.0, 1.0) + 1.0) / 2.0

        time_in_phase_norm = min(self._time_in_phase / self.MAX_PHASE_T, 1.0)

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

    def _get_reward(self, switched, wasted_vote):
        total_queue = 0.0
        total_wait = 0.0
        n_lanes = 0

        for axis, group in self.INCOMING_LANES.items():
            for lane in group:
                q = self.conn.lane.getLastStepHaltingNumber(lane)
                w = self.conn.lane.getWaitingTime(lane)
                total_queue += q
                total_wait += w
                n_lanes += 1

        queue_penalty = -(total_queue / n_lanes) / self.max_queue
        wait_penalty  = -(total_wait / n_lanes) / self.MAX_WAIT
        reward = 0.7 * queue_penalty + 0.3 * wait_penalty

        if self.imbalance_bonus_weight > 0 and not self._in_yellow:
            ns_q, ew_q = self._get_raw_ns_ew_queue()
            raw_imbalance = ns_q - ew_q
            on_ns = (self._phase == 0)
            signed_alignment = raw_imbalance if on_ns else -raw_imbalance
            reward += self.imbalance_bonus_weight * (signed_alignment / self.max_queue)

        if switched:
            reward -= self.switch_penalty
        if wasted_vote:
            reward -= self.wasted_vote_penalty
        if self._wrong_direction_this_step:
            reward -= self.wrong_direction_penalty

        return reward

    def _apply_action(self, action):
        switched = False
        wasted_vote = False
        self._last_hold_duration = None
        self._wrong_direction_this_step = False

        if self._in_yellow:
            if action == 1:
                wasted_vote = True
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
            eligible = self._time_in_phase >= self.MIN_GREEN

            if action == 1 and not eligible and not force_switch:
                wasted_vote = True

            if (action == 1 and eligible) or force_switch:
                wrong_direction = False
                if action == 1 and not force_switch:
                    ns_q, ew_q = self._get_raw_ns_ew_queue()
                    currently_on_ns = (self._phase == 0)
                    leaving_busier_side = (currently_on_ns and ns_q > ew_q) or \
                                          (not currently_on_ns and ew_q > ns_q)
                    wrong_direction = leaving_busier_side

                self._last_hold_duration = self._time_in_phase
                yellow_phase = self.PHASE_NS_YELLOW if self._phase == 0 else self.PHASE_EW_YELLOW
                self.conn.trafficlight.setPhase(self.TL_ID, yellow_phase)
                self.conn.trafficlight.setPhaseDuration(self.TL_ID, 9999)
                self._in_yellow      = True
                self._time_in_yellow = 0
                switched = (action == 1)
                self._wrong_direction_this_step = wrong_direction

        return switched, wasted_vote

    # ---------------------------------------------------------------- #
    # gym API
    # ---------------------------------------------------------------- #

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
        self._phase          = 0
        self._time_in_phase  = 0
        self._in_yellow      = False
        self._time_in_yellow = 0
        self._last_hold_duration = None
        self._wrong_direction_this_step = False

        self._start_sumo()
        self.conn.trafficlight.setPhase(self.TL_ID, self.PHASE_NS_GREEN)
        self.conn.trafficlight.setPhaseDuration(self.TL_ID, 9999)
        self.conn.simulationStep()

        return self._get_obs(), {}

    def step(self, action):
        switched, wasted_vote = self._apply_action(int(action))
        self.conn.simulationStep()
        self._step_count += 1

        obs    = self._get_obs()
        reward = self._get_reward(switched, wasted_vote)
        done   = self._step_count >= self.max_steps

        ns_q, ew_q = self._get_raw_ns_ew_queue()
        info = {
            "local_queue": self._get_local_queue(),
            "switched": switched,
            "wasted_vote": wasted_vote,
            "ns_queue": ns_q,
            "ew_queue": ew_q,
            "imbalance": ns_q - ew_q,
            "phase": self._phase,
            "hold_duration_at_switch": self._last_hold_duration,
            "wrong_direction": self._wrong_direction_this_step,
            "sumo_seed": self._seed,
        }

        return obs, reward, done, False, info

    def close(self):
        self._close_sumo()

    def render(self, mode="human"):
        pass