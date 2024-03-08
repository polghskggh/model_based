from abc import abstractmethod

import numpy as np


class CriticInterface:
    def __init__(self):
        pass

    @abstractmethod
    def update_model(self, reward: np.ndarray[float], state: np.ndarray[float], action: np.ndarray[float],
                     next_state: np.ndarray[float], next_action: np.ndarray[float]):
        pass

    @abstractmethod
    def provide_feedback(self, state: np.ndarray[float], action: np.ndarray[float]):
        pass
