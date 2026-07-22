"""
wrappers.py

Action-space adapter so SB3's DQN (Discrete-only) can train on the same
SumoTrafficEnv2J that PPO trains on directly (MultiDiscrete([2, 2])).

This is the ONLY thing that differs about how DQN sees the environment.
Observation space, reward function, episode length, seed rotation, and
every dynamic in env.py are completely untouched -- so PPO and DQN are
optimizing the exact same MDP, just through different action encodings.
That's what makes the eventual comparison meaningful rather than
"PPO on env A vs DQN on a different env B".
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlattenMultiDiscreteAction(gym.ActionWrapper):
    """
    MultiDiscrete([2, 2]) -> Discrete(4).

    SB3's DQN only supports a single Discrete action space (one Q-value
    per action index, from one output head). SumoTrafficEnv2J exposes
    two independent binary switch-votes (one per junction), so we
    flatten the 2x2 combinations into 4 indices:

        action_idx -> [vote_J1, vote_J2]
              0     ->   [0, 0]     hold both
              1     ->   [0, 1]     hold J1, vote switch J2
              2     ->   [1, 0]     vote switch J1, hold J2
              3     ->   [1, 1]     vote switch both

    Order matches env.SumoTrafficEnv2J.TL_IDS = ["J1", "J2"].

    NOTE ON WHAT THIS DOES *NOT* CHANGE: this wrapper only rewrites how
    an action integer is turned into the [a_J1, a_J2] pair the base env
    already expects in step(). Reward, observation, info dict content,
    episode length, MIN_GREEN/MAX_GREEN logic, wasted-vote/switch
    penalties -- all untouched, because they all live downstream of
    this wrapper in the base env.
    """

    LUT = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.MultiDiscrete), (
            "FlattenMultiDiscreteAction expects the raw SumoTrafficEnv2J "
            f"MultiDiscrete action space, got {type(env.action_space)}"
        )
        assert list(env.action_space.nvec) == [2, 2], (
            f"Expected MultiDiscrete([2, 2]), got nvec={list(env.action_space.nvec)}"
        )
        self.action_space = spaces.Discrete(4)

    def action(self, act):
        return self.LUT[int(act)]

    def reverse_action(self, act):
        # Not needed for training/eval (SB3 never calls this), provided
        # only in case you want to convert a [a_J1, a_J2] pair back to
        # a Discrete index somewhere in analysis code.
        for idx, pair in enumerate(self.LUT):
            if tuple(pair) == tuple(act):
                return idx
        raise ValueError(f"action pair {act} not in LUT")