from abc import abstractmethod

import numpy as np


class ActorInterface:
    def __init__(self):
        pass

    @abstractmethod
    def approximate_best_action(self, state: np.ndarray[float]) -> np.ndarray[float]:
        pass

    @abstractmethod
    def update_model(self, state: np.ndarray[float], selected_actions: np.ndarray[float],
                     action_grads: np.ndarray[float]):
        pass

    @abstractmethod
    def calculate_actions(self, new_states: np.ndarray[float]) -> np.ndarray[float]:
        pass
