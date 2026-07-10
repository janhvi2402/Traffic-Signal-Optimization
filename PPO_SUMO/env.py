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
        1 = REQUEST a phase switch. This is only honoured once the
            junction has held its current phase for >= MIN_GREEN
            seconds (or forced once MAX_GREEN is hit). Voting 1 too
            early is silently ignored by the env, which is why a
            policy that always votes 1 can look "fine" reward-wise
            unless you explicitly penalise the switch itself (see
            SWITCH_PENALTY below).

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
    -(mean queue length across all incoming lanes of both junctions,
      normalised by max_queue)
    - SWITCH_PENALTY   * (number of junctions that just initiated a
      yellow transition this step)
    - WASTED_VOTE_PENALTY * (number of junctions that voted action=1
      while ineligible — i.e. currently mid-yellow, or before
      MIN_GREEN has elapsed)

    WHY the wasted-vote term exists: previously, voting 1 while
    ineligible cost NOTHING — the env just silently discarded the
    vote. That made "always vote 1" a strictly dominant strategy: it
    never cost more than voting 0 until eligible then voting 1, so
    the policy never had a reason to actually condition on the queue
    features. Diagnostics on a trained model showed exactly this —
    277/277 switches on J1 and J2 landing on the identical step, at a
    period matching MIN_GREEN + YELLOW_TIME exactly, i.e. the agent
    voting 1 every single step for both junctions regardless of their
    own state. Penalizing wasted votes closes that loophole; the
    SWITCH_PENALTY term alone wasn't enough because it only fires
    when a vote actually gets honored, not when it's spammed for free
    in between.
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
    # MAX_QUEUE was a flat guess (30 veh/lane) before. If your actual
    # queues rarely exceed ~8-10 veh/lane (check by logging raw
    # getLastStepHaltingNumber() over a random-policy rollout before
    # normalising), the observation spends its whole life squeezed
    # into [0, 0.3] and the network has very little dynamic range to
    # key decisions off. Recalibrate this from real data if you can;
    # in the meantime it's exposed as a constructor arg instead of a
    # hardcoded class constant so you can sweep it without editing
    # the file.
    MAX_QUEUE_DEFAULT = 30    # vehicles per lane  (for normalisation)
    MAX_WAIT    = 120   # seconds            (for normalisation)
    MAX_PHASE_T = 60    # seconds            (for normalisation)
    MIN_GREEN   = 10    # minimum green duration before a switch is allowed
    MAX_GREEN   = 90    # seconds — hard cap so a phase can't run forever
    YELLOW_TIME = 3      # seconds — matches the net file's yellow duration

    # Cost of actually initiating a phase switch (per junction, per
    # switch event). Tune this relative to the queue term: reward is
    # in roughly [-1, 0] per step from queue alone. Raised 0.05 -> 0.15
    # because 0.05 wasn't enough on its own to stop the "always
    # switch" degenerate policy — see WASTED_VOTE_PENALTY below, which
    # addresses the actual root cause.
    SWITCH_PENALTY = 0.15

    # Cost of voting action=1 while the vote can't be honored yet
    # (mid-yellow, or before MIN_GREEN has elapsed). This is the key
    # fix: without it, spamming action=1 every step is free until the
    # moment it's actually acted on, so the policy has no incentive to
    # ever condition on the queue features — it can just always vote 1
    # and let the MIN_GREEN gate do the work. This penalty makes idle
    # spamming cost something, forcing the policy to only vote 1 when
    # it actually wants a switch soon.
    WASTED_VOTE_PENALTY = 0.03

    def __init__(
        self,
        cfg_path    = None,
        use_gui     = False,
        max_steps   = 3600,
        seed        = None,
        port=8813,
        max_queue   = None,       # override MAX_QUEUE_DEFAULT if you've calibrated it
        switch_penalty = None,    # override SWITCH_PENALTY if you want to sweep it
        wasted_vote_penalty = None,  # override WASTED_VOTE_PENALTY if you want to sweep it
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
        self.max_queue = max_queue if max_queue is not None else self.MAX_QUEUE_DEFAULT
        self.switch_penalty = switch_penalty if switch_penalty is not None else self.SWITCH_PENALTY
        self.wasted_vote_penalty = (
            wasted_vote_penalty if wasted_vote_penalty is not None else self.WASTED_VOTE_PENALTY
        )

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
        ]
        # FIX: --random and --seed are contradictory (--random tells SUMO
        # to pick its own seed from system time, overriding yours). Only
        # pass one or the other so per-episode seeds actually take effect.
        if self._seed is not None:
            safe_seed = int(self._seed) % 2_147_483_647
            cmd += ["--seed", str(safe_seed)]
        else:
            cmd += ["--random"]

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

            obs.append(np.mean(ns_q) / self.max_queue)
            obs.append(np.mean(ew_q) / self.max_queue)
            obs.append(min(np.mean(ns_w) / self.MAX_WAIT, 1.0))
            obs.append(min(np.mean(ew_w) / self.MAX_WAIT, 1.0))
            obs.append(float(self._phase[tl]))                               # 0 or 1
            obs.append(min(self._time_in_phase[tl] / self.MAX_PHASE_T, 1.0))
            obs.append(float(self._in_yellow[tl]))

        return np.clip(np.array(obs, dtype=np.float32), 0.0, 1.0)

    def _get_local_queue(self, tl):
        """Mean queue length across ONE junction's incoming lanes (raw, not normalised).
        Exposed for diagnostics — lets you check post-fix whether J1 and J2
        actually start behaving differently instead of mirroring each other."""
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
        return queue_term + switch_term + wasted_term

    def _apply_action(self, tl, action):
        """Returns (switched, wasted_vote):
          switched     - True if this call just initiated a yellow transition
                          (i.e. a switch actually happened, not just a vote).
          wasted_vote  - True if action==1 was cast but couldn't be honored
                          this step (mid-yellow, or MIN_GREEN not yet met).
                          This is what used to be free; now it's penalized.
        """
        switched = False
        wasted_vote = False

        if self._in_yellow[tl]:
            # Vote is moot — we're already committed to this transition.
            if action == 1:
                wasted_vote = True

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
            eligible = self._time_in_phase[tl] >= self.MIN_GREEN

            if action == 1 and not eligible and not force_switch:
                # Voted to switch too early — this used to be silently
                # discarded for free. Now it costs a little, so
                # spamming action=1 every step is no longer a free ride.
                wasted_vote = True

            if (action == 1 and eligible) or force_switch:
                yellow_phase = self.PHASE_NS_YELLOW if self._phase[tl] == 0 else self.PHASE_EW_YELLOW
                self.conn.trafficlight.setPhase(tl, yellow_phase)
                self.conn.trafficlight.setPhaseDuration(tl, 9999)   # prevent auto-advance
                self._in_yellow[tl]      = True
                self._time_in_yellow[tl] = 0
                # Only charge the switch penalty for agent-requested switches,
                # not ones forced by MAX_GREEN — the agent shouldn't be
                # punished for a switch it didn't choose.
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

        self._start_sumo()

        for tl in self.TL_IDS:
            self.conn.trafficlight.setPhase(tl, self.PHASE_NS_GREEN)
            self.conn.trafficlight.setPhaseDuration(tl, 9999)   # prevent auto-advance from the very first step

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

        # Per-junction diagnostics — doesn't affect training, just lets you
        # plot/inspect J1 vs J2 behaviour separately after the fact.
        info = {
            "local_queue": {tl: self._get_local_queue(tl) for tl in self.TL_IDS},
            "switched": switches,
            "wasted_vote": wasted_votes,
        }

        return obs, reward, done, False, info

    def close(self):
        self._close_sumo()

    def render(self, mode="human"):
        # GUI is enabled at construction time via use_gui=True;
        # this method is a no-op for headless runs.
        pass