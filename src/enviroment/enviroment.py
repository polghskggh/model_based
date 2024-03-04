from typing import Tuple, Any, Dict, SupportsFloat

import gymnasium as gym
import numpy as np


class Enviroment:
    def __init__(self):
        self.enviroment: gym.Env = gym.make("ALE/Breakout-v5", render_mode="human")

    def reset(self) -> tuple[Any, dict[str, Any]]:
        return self.enviroment.reset()

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        action = np.argmax(action)
        return self.enviroment.step(action)

    def close(self):
        self.enviroment.close()