import gymnasium as gym
import numpy as np
from gymnasium import spaces
#We use Gymnasium because it provides a standard interface, and PPO is written to work with environments that follow that interface

class TrafficEnv3J(gym.Env):
    """
    3 junctions in a line: J0 -- J1 -- J2

    Inter-junction flow:
      - vehicles leaving J0 east  arrive at J1 west after travel_delay steps
      - vehicles leaving J1 east  arrive at J2 west after travel_delay steps
      - vehicles leaving J1 west  arrive at J0 east after travel_delay steps
      - vehicles leaving J2 west  arrive at J1 east after travel_delay steps

    Each junction has 4 queues: [N, S, E, W]
    Phase 0 = N-S green, Phase 1 = E-W green
    """

    def __init__(self, max_queue=20, max_steps=500,
             yellow_duration=3, travel_delay=5,
             arrival_rates=None, randomize_rates=False):
        super().__init__()
        self.max_queue = max_queue
        self.max_steps = max_steps
        self.yellow_duration = yellow_duration
        self.travel_delay = travel_delay
        self.n_junctions = 3
        self.randomize_rates = randomize_rates   # flag to control this

        # fixed rates if provided, else default
        if arrival_rates is None:
            self.arrival_rates = {
                0: [0.4, 0.4, 0.1, 0.1],
                1: [0.4, 0.4, 0.1, 0.1],
                2: [0.4, 0.4, 0.1, 0.1],
            }
        else:
            self.arrival_rates = arrival_rates

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(21,), dtype=np.float32 #3 junctions × 7 features = 21 
        )
        self.action_space = spaces.MultiDiscrete([2, 2, 2])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize_rates:
            for j in range(self.n_junctions):
                ns = self.np_random.uniform(0.1, 0.7)
                ew = self.np_random.uniform(0.1, 0.7)
                self.arrival_rates[j] = [ns, ns, ew, ew]

        self.queues = np.zeros((3, 4), dtype=np.float32)
        self.phase = [0, 0, 0]
        self.time_in_phase = [0, 0, 0]
        self.in_yellow = [False, False, False]
        self.yellow_timer = [0, 0, 0]
        self.step_count = 0
        self.pipeline = [[], [], [], []]

        return self._get_obs(), {}

        self.queues = np.zeros((3, 4), dtype=np.float32)
        self.phase = [0, 0, 0]
        self.time_in_phase = [0, 0, 0]
        self.in_yellow = [False, False, False]
        self.yellow_timer = [0, 0, 0]
        self.step_count = 0
        self.pipeline = [[], [], [], []]

        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for j in range(self.n_junctions):
            for q in range(4):
                obs.append(self.queues[j][q] / self.max_queue)
            obs.append(float(self.phase[j]))
            obs.append(min(self.time_in_phase[j] / 60.0, 1.0))
            obs.append(float(self.in_yellow[j]))
        return np.array(obs, dtype=np.float32)

    def _arrival_step(self):
        for j in range(self.n_junctions):
            for i, rate in enumerate(self.arrival_rates[j]):
                arrivals = self.np_random.poisson(rate)
                self.queues[j][i] = min(
                    self.queues[j][i] + arrivals, self.max_queue
                )

    def _departure_step(self):
        """
        Vehicles depart based on active green phase.
        Vehicles leaving on the E-W axis enter the inter-junction pipeline.
        Returns dict of inter-junction flows generated this step.
        """
        sat_flow = 1.8
        flows = {}   # key: pipeline index, value: volume transferred

        for j in range(self.n_junctions):
            if self.in_yellow[j]:
                continue

            if self.phase[j] == 0:   # N-S green
                self.queues[j][0] = max(0, self.queues[j][0] - sat_flow)  # N
                self.queues[j][1] = max(0, self.queues[j][1] - sat_flow)  # S
            else:                    # E-W green
                departed_e = min(self.queues[j][2], sat_flow)
                departed_w = min(self.queues[j][3], sat_flow)
                self.queues[j][2] = max(0, self.queues[j][2] - sat_flow)  # E
                self.queues[j][3] = max(0, self.queues[j][3] - sat_flow)  # W

                # vehicles leaving east of J0 go to west of J1
                if j == 0 and departed_e > 0:
                    flows[0] = departed_e
                # vehicles leaving east of J1 go to west of J2
                if j == 1 and departed_e > 0:
                    flows[1] = departed_e
                # vehicles leaving west of J1 go to east of J0
                if j == 1 and departed_w > 0:
                    flows[2] = departed_w
                # vehicles leaving west of J2 go to east of J1
                if j == 2 and departed_w > 0:
                    flows[3] = departed_w

        return flows

    def _update_pipeline(self, new_flows):
        """Tick down travel delay counters, deliver arrived vehicles."""

        # add new flows into pipeline
        for pipe_idx, volume in new_flows.items():
            self.pipeline[pipe_idx].append([self.travel_delay, volume])

        # tick and deliver
        # pipe 0: J0-east -> J1-west (queue index 3)
        # pipe 1: J1-east -> J2-west (queue index 3)
        # pipe 2: J1-west -> J0-east (queue index 2)
        # pipe 3: J2-west -> J1-east (queue index 2)
        destinations = [
            (1, 3),   # pipe 0 -> junction 1, queue W
            (2, 3),   # pipe 1 -> junction 2, queue W
            (0, 2),   # pipe 2 -> junction 0, queue E
            (1, 2),   # pipe 3 -> junction 1, queue E
        ]

        for pipe_idx, (dest_j, dest_q) in enumerate(destinations):
            still_traveling = []
            for entry in self.pipeline[pipe_idx]:
                entry[0] -= 1
                if entry[0] <= 0:
                    # arrived — add to destination queue
                    self.queues[dest_j][dest_q] = min(
                        self.queues[dest_j][dest_q] + entry[1],
                        self.max_queue
                    )
                else:
                    still_traveling.append(entry)
            self.pipeline[pipe_idx] = still_traveling

    def step(self, action):
        # action is array [a0, a1, a2] one per junction
        for j in range(self.n_junctions):
            if self.in_yellow[j]:
                self.yellow_timer[j] += 1
                if self.yellow_timer[j] >= self.yellow_duration:
                    self.in_yellow[j] = False
                    self.phase[j] = 1 - self.phase[j]
                    self.time_in_phase[j] = 0
                    self.yellow_timer[j] = 0
            else:
                if action[j] == 1:
                    self.in_yellow[j] = True
                    self.yellow_timer[j] = 0
                else:
                    self.time_in_phase[j] += 1

        self._arrival_step()
        new_flows = self._departure_step()
        self._update_pipeline(new_flows)

        # reward: negative mean queue across all junctions, normalized
        reward = -float(np.mean(self.queues)) / self.max_queue

        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self._get_obs(), reward, done, False, {}