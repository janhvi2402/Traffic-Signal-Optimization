import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TrafficEnv(gym.Env):
    def __init__(self, max_queue=20, max_steps=500, yellow_duration=3):
        super().__init__()
        self.max_queue = max_queue
        self.max_steps = max_steps
        self.yellow_duration = yellow_duration

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(      # ✅ moved here, inside __init__
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.queues = np.zeros(4, dtype=np.float32)
        self.waiting_time = np.zeros(4, dtype=np.float32)
        self.phase = 0
        self.time_in_phase = 0
        self.step_count = 0
        self.in_yellow = False
        self.yellow_timer = 0
        return self._get_obs(), {}
        # ✅ removed the observation_space definition that was here (after return = unreachable)

    def _get_obs(self):                           # ✅ indented inside the class
        max_wait = 200.0
        return np.array([
            self.queues[0] / self.max_queue,
            self.queues[1] / self.max_queue,
            self.queues[2] / self.max_queue,
            self.queues[3] / self.max_queue,
            float(self.phase),
            min(self.time_in_phase / 60.0, 1.0),
            min(self.waiting_time[0] / max_wait, 1.0),
            min(self.waiting_time[1] / max_wait, 1.0),
            min(self.waiting_time[2] / max_wait, 1.0),
            min(self.waiting_time[3] / max_wait, 1.0),
        ], dtype=np.float32)

    def _arrival_step(self):
        rates = [0.4, 0.4, 0.3, 0.3]
        for i, rate in enumerate(rates):
            arrivals = self.np_random.poisson(rate)
            self.queues[i] = min(self.queues[i] + arrivals, self.max_queue)

    def _departure_step(self):
        if self.in_yellow:
            return
        sat_flow = 1.8
        if self.phase == 0:
            self.queues[0] = max(0, self.queues[0] - sat_flow)
            self.queues[1] = max(0, self.queues[1] - sat_flow)
        else:
            self.queues[2] = max(0, self.queues[2] - sat_flow)
            self.queues[3] = max(0, self.queues[3] - sat_flow)

    def _update_waiting(self):
        if self.phase == 0:
            self.waiting_time[0] = max(0, self.waiting_time[0] - 1)
            self.waiting_time[1] = max(0, self.waiting_time[1] - 1)
            self.waiting_time[2] += self.queues[2]
            self.waiting_time[3] += self.queues[3]
        else:
            self.waiting_time[2] = max(0, self.waiting_time[2] - 1)
            self.waiting_time[3] = max(0, self.waiting_time[3] - 1)
            self.waiting_time[0] += self.queues[0]
            self.waiting_time[1] += self.queues[1]

    def step(self, action):
        if self.in_yellow:
            self.yellow_timer += 1
            if self.yellow_timer >= self.yellow_duration:
                self.in_yellow = False
                self.phase = 1 - self.phase
                self.time_in_phase = 0
                self.yellow_timer = 0
        else:
            if action == 1:
                self.in_yellow = True
                self.yellow_timer = 0
            else:
                self.time_in_phase += 1

        self._arrival_step()
        self._departure_step()
        self._update_waiting()

        queue_penalty   = np.sum(self.queues) / (self.max_queue * 4)          # 0 to 1
        waiting_penalty = np.sum(self.waiting_time) / (200.0 * 4)             # 0 to 1, 200 = max_wait
        reward = -(queue_penalty + 0.3 * waiting_penalty)                     # total range roughly 0 to -1.3
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self._get_obs(), reward, done, False, {}