from abc import abstractmethod

import numpy as np

from src.agent.actor.actorinterface import ActorInterface


class CriticInterface:
    def __init__(self):
        pass

    @abstractmethod
    def calculate_grads(self, *args):
        pass

    @abstractmethod
    def provide_feedback(self, *args):
        pass

    @abstractmethod
    def update(self, grads: dict):
        pass

