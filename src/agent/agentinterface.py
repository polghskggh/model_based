from abc import abstractmethod

import numpy as np

from src.pod.replaybuffer import ReplayBuffer


class AgentInterface:
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

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def last_transition(self):
        pass
