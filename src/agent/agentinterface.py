from abc import abstractmethod

import numpy as np


class AgentInterface:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def update_policy(self):
        pass

    @abstractmethod
    def select_action(self) -> np.ndarray[float]:
        pass

    @abstractmethod
    def receive_reward(self, reward: float):
        pass

    @abstractmethod
    def receive_state(self, state: np.ndarray[float]):
        pass
