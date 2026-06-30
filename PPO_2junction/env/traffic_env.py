import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque


class TrafficEnv2J(gym.Env):
    """
    2-junction linear network with correct lane topology.

    Topology:
          N1        N2
          |          |
    W1 — J1 ——————  J2 — E2
          |          |
          S1        S2

    J1 legs : W1 (west, external), N1 (north, external),
              S1 (south, external), J2 (east, inter-junction)
    J2 legs : E2 (east, external), N2 (north, external),
              S2 (south, external), J1 (west, inter-junction)

    Each leg has TWO directional queues:
        - inbound  : vehicles arriving at the junction from that leg
        - outbound : vehicles leaving the junction toward that leg

    Phases per junction (2-phase signalised control):
        Phase 0 — N↔S green  (N-inbound and S-inbound discharge)
        Phase 1 — E↔W green  (E-inbound and W-inbound discharge)

    Inter-junction transfer:
        J1 east-outbound  →  travel pipeline  →  J2 west-inbound
        J2 west-outbound  →  travel pipeline  →  J1 east-inbound
    """

    #  Lane index constants — makes the rest of the code self-documenting  #
    # Inbound queue indices (vehicles heading INTO the junction)
    N_IN  = 0   # from N-arm  toward junction
    S_IN  = 1   # from S-arm  toward junction
    E_IN  = 2   # from E-arm  toward junction
    W_IN  = 3   # from W-arm  toward junction

    # Outbound queue indices (vehicles leaving the junction)
    N_OUT = 4   # junction  →  N-arm
    S_OUT = 5   # junction  →  S-arm
    E_OUT = 6   # junction  →  E-arm  (J1: toward J2 | J2: toward E2)
    W_OUT = 7   # junction  →  W-arm  (J1: toward W1 | J2: toward J1)

    N_QUEUES = 8  # per junction

    def __init__(
        self,
        max_queue      = 20,
        max_steps      = 500,
        yellow_duration= 3,
        travel_delay   = 5,
        arrival_rates  = None,
        randomize_rates= False,
        sat_flow       = 1.8,
    ):
        super().__init__()

        self.max_queue       = max_queue
        self.max_steps       = max_steps
        self.yellow_duration = yellow_duration
        self.travel_delay    = travel_delay
        self.n_junctions     = 2
        self.randomize_rates = randomize_rates
        self.sat_flow        = sat_flow


        # External arrival rates — only INBOUND external legs get arrivals.
        #   J1 external inbound legs : N_IN, S_IN, W_IN
        #   J2 external inbound legs : N_IN, S_IN, E_IN
        #
        # Format: { junction_id: [N_IN, S_IN, E_IN, W_IN] }
        # (E_IN for J1 = from J2 via pipeline, so external rate = 0)
        # (W_IN for J2 = from J1 via pipeline, so external rate = 0)

        if arrival_rates is None:
            self.base_arrival_rates = {
                0: [0.4, 0.4, 0.0, 0.2],   # J1: N, S, E(inter), W(ext)
                1: [0.4, 0.4, 0.2, 0.0],   # J2: N, S, E(ext),   W(inter)
            }
        else:
            self.base_arrival_rates = arrival_rates

        # Observation: per junction — 8 queue lengths + phase + time_in_phase
        #              + in_yellow = 11 features × 2 junctions = 22
        obs_dim = self.n_junctions * (self.N_QUEUES + 3)  # 8 queues + 3 scalars
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([2, 2])

    #  Reset                                                               
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Optionally randomise arrival rates for curriculum / generalisation
        if self.randomize_rates:
            self.arrival_rates = {}
            for j in range(self.n_junctions):
                ns = float(self.np_random.uniform(0.1, 0.6))
                ew = float(self.np_random.uniform(0.05, 0.3))
                rates = [ns, ns, 0.0, ew] if j == 0 else [ns, ns, ew, 0.0]
                self.arrival_rates[j] = rates
        else:
            self.arrival_rates = {
                j: list(self.base_arrival_rates[j])
                for j in range(self.n_junctions)
            }

        # queues[j][q] — one row per junction, N_QUEUES columns
        self.queues       = np.zeros((self.n_junctions, self.N_QUEUES), dtype=np.float32)
        self.phase        = [0, 0]
        self.time_in_phase= [0, 0]
        self.in_yellow    = [False, False]
        self.yellow_timer = [0, 0]
        self.step_count   = 0

        # Two inter-junction pipelines:
        #   pipeline[0] : J1 E_OUT  →  J2 W_IN   (J1 → J2)
        #   pipeline[1] : J2 W_OUT  →  J1 E_IN   (J2 → J1)
        # Each entry is [remaining_delay, volume]
        self.pipeline = [[], []]

        return self._get_obs(), {}

    #  Observation                                                         
    def _get_obs(self):
        obs = []
        for j in range(self.n_junctions):
            for q in range(self.N_QUEUES):
                obs.append(self.queues[j][q] / self.max_queue)
            obs.append(float(self.phase[j]))
            obs.append(min(self.time_in_phase[j] / 60.0, 1.0))
            obs.append(float(self.in_yellow[j]))
        return np.array(obs, dtype=np.float32)

    #  Arrival step — external demand only                                 
    def _arrival_step(self):
        for j in range(self.n_junctions):
            for lane_idx in [self.N_IN, self.S_IN, self.E_IN, self.W_IN]:
                rate = self.arrival_rates[j][lane_idx]
                if rate > 0:
                    arrivals = self.np_random.poisson(rate)
                    self.queues[j][lane_idx] = min(
                        self.queues[j][lane_idx] + arrivals,
                        self.max_queue
                    )

    #  Departure step — green-phase vehicles discharge                     
    def _departure_step(self):
        """
        Returns inter-junction flows:
            flows[0] = volume dispatched from J1 → J2  (J1 E_OUT)
            flows[1] = volume dispatched from J2 → J1  (J2 W_OUT)
        """
        inter_flows = {}

        for j in range(self.n_junctions):
            if self.in_yellow[j]:
                continue  # no discharge during yellow

            if self.phase[j] == 0:
                # ---- N↔S green ----
                # N-inbound discharges → vehicles exit via S-outbound (through-move)
                # S-inbound discharges → vehicles exit via N-outbound (through-move)
                # (simplified: assume through movement; turning can be added later)

                dep_n = min(self.queues[j][self.N_IN], self.sat_flow)
                dep_s = min(self.queues[j][self.S_IN], self.sat_flow)

                self.queues[j][self.N_IN]  = max(0.0, self.queues[j][self.N_IN]  - dep_n)
                self.queues[j][self.S_IN]  = max(0.0, self.queues[j][self.S_IN]  - dep_s)

                # Through: N→S and S→N (outbound queues accumulate momentarily
                # then drain each step — treated as pass-through here)
                self.queues[j][self.S_OUT] = max(0.0, self.queues[j][self.S_OUT] + dep_n - self.sat_flow)
                self.queues[j][self.N_OUT] = max(0.0, self.queues[j][self.N_OUT] + dep_s - self.sat_flow)

            else:
                # ---- E↔W green ----
                dep_e = min(self.queues[j][self.E_IN], self.sat_flow)
                dep_w = min(self.queues[j][self.W_IN], self.sat_flow)

                self.queues[j][self.E_IN]  = max(0.0, self.queues[j][self.E_IN]  - dep_e)
                self.queues[j][self.W_IN]  = max(0.0, self.queues[j][self.W_IN]  - dep_w)

                if j == 0:
                    # J1 E↔W green:
                    #   W_IN (from W1) discharges → exits E_OUT toward J2
                    #   E_IN (from J2) discharges → exits W_OUT toward W1
                    j1_to_j2 = dep_w   # W1 traffic travels through J1 eastward
                    j1_to_w1 = dep_e   # J2 traffic travels through J1 westward

                    self.queues[j][self.E_OUT] = max(0.0, self.queues[j][self.E_OUT] + j1_to_j2 - self.sat_flow)
                    self.queues[j][self.W_OUT] = max(0.0, self.queues[j][self.W_OUT] + j1_to_w1 - self.sat_flow)

                    if j1_to_j2 > 0:
                        inter_flows[0] = j1_to_j2   # J1 → J2

                else:
                    # J2 E↔W green:
                    #   E_IN (from E2) discharges → exits W_OUT toward J1
                    #   W_IN (from J1) discharges → exits E_OUT toward E2
                    j2_to_j1 = dep_e   # E2 traffic travels through J2 westward
                    j2_to_e2 = dep_w   # J1 traffic travels through J2 eastward

                    self.queues[j][self.W_OUT] = max(0.0, self.queues[j][self.W_OUT] + j2_to_j1 - self.sat_flow)
                    self.queues[j][self.E_OUT] = max(0.0, self.queues[j][self.E_OUT] + j2_to_e2 - self.sat_flow)

                    if j2_to_j1 > 0:
                        inter_flows[1] = j2_to_j1   # J2 → J1

        return inter_flows

    #  Pipeline — inter-junction travel delay                              
    def _update_pipeline(self, new_flows):
        """
        pipeline[0]: J1 E_OUT → (travel_delay steps) → J2 W_IN
        pipeline[1]: J2 W_OUT → (travel_delay steps) → J1 E_IN
        """
        destinations = [
            (1, self.W_IN),   # pipeline 0 arrives at J2 W_IN
            (0, self.E_IN),   # pipeline 1 arrives at J1 E_IN
        ]

        # Inject new flows
        for pipe_idx, volume in new_flows.items():
            self.pipeline[pipe_idx].append([self.travel_delay, volume])

        # Tick and deliver
        for pipe_idx, (dest_j, dest_q) in enumerate(destinations):
            still_traveling = []
            for entry in self.pipeline[pipe_idx]:
                entry[0] -= 1
                if entry[0] <= 0:
                    self.queues[dest_j][dest_q] = min(
                        self.queues[dest_j][dest_q] + entry[1],
                        self.max_queue
                    )
                else:
                    still_traveling.append(entry)
            self.pipeline[pipe_idx] = still_traveling

    #  Step                                                                
    def step(self, action):
        # --- Phase / yellow logic ---
        for j in range(self.n_junctions):
            if self.in_yellow[j]:
                self.yellow_timer[j] += 1
                if self.yellow_timer[j] >= self.yellow_duration:
                    self.in_yellow[j]     = False
                    self.phase[j]         = 1 - self.phase[j]
                    self.time_in_phase[j] = 0
                    self.yellow_timer[j]  = 0
            else:
                if action[j] == 1:        # agent requests phase switch
                    self.in_yellow[j]    = True
                    self.yellow_timer[j] = 0
                else:
                    self.time_in_phase[j] += 1

        # --- Traffic dynamics ---
        self._arrival_step()
        new_flows = self._departure_step()
        self._update_pipeline(new_flows)

        # --- Reward: penalise mean inbound queue congestion ---
        inbound_queues = self.queues[:, [self.N_IN, self.S_IN, self.E_IN, self.W_IN]]
        reward = -float(np.mean(inbound_queues)) / self.max_queue

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self._get_obs(), reward, done, False, {}

    #  Optional: human-readable queue summary 
    def render(self, mode="human"):
        labels = ["N_in", "S_in", "E_in", "W_in", "N_out", "S_out", "E_out", "W_out"]
        for j in range(self.n_junctions):
            phase_str = "NS-green" if self.phase[j] == 0 else "EW-green"
            yellow_str = " [YELLOW]" if self.in_yellow[j] else ""
            print(f"\nJ{j+1} | {phase_str}{yellow_str} | t={self.time_in_phase[j]}")
            for i, lbl in enumerate(labels):
                print(f"  {lbl:8s}: {self.queues[j][i]:.1f}")