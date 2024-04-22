from abc import abstractmethod

import numpy as np

from src.models.modelwrapper import ModelWrapper


class ActorInterface:
    @abstractmethod
    def approximate_best_action(self, state: np.ndarray[float]) -> np.ndarray[float]:
        pass

    @abstractmethod
    def calculate_actions(self, new_states: np.ndarray[float]) -> np.ndarray[float]:
        pass

    @abstractmethod
    def update(self, *args):
        pass

    @abstractmethod
    def model(self) -> ModelWrapper:
        pass
