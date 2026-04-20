import os
from typing import Dict

import gym
import numpy as np
from soccer_twos import AgentInterface


class HandsOnAgent(AgentInterface):
    """A baseline agent implementation for the hands-on assignment."""

    def __init__(self, env: gym.Env):
        super().__init__()
        self.action_space = env.action_space
        self.model_path = os.path.join(os.path.dirname(__file__), "checkpoint.pth")

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """Return one action per team player using the environment action space."""
        actions = {}
        for player_id in observation:
            actions[player_id] = self.action_space.sample()
        return actions
