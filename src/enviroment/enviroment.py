from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


class Enviroment:
    def __init__(self):
        self.enviroment: gym.Env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

    def reset(self) -> tuple[Any, dict[str, Any]]:
        return self.enviroment.reset()

    def step(self, action) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        action = np.argmax(action)
        return self.enviroment.step(action)

    def sample_action(self) -> int:
        return self.enviroment.action_space.sample()


    def close(self):
        self.enviroment.close()
