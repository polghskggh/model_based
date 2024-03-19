from abc import abstractmethod

import numpy as np

from src.agent.actor.actorinterface import ActorInterface


class CriticInterface:
    def __init__(self):
        pass

    @abstractmethod
    def calculate_grads(self, reward: np.ndarray[float], state: np.ndarray[float], action: np.ndarray[float],
                        next_state: np.ndarray[float], next_action: np.ndarray[float]):
        pass

    @abstractmethod
    def provide_feedback(self, actor: ActorInterface, states: np.ndarray[float]):
        pass

    @abstractmethod
    def update(self):
        pass

