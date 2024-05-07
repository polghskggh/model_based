from abc import abstractmethod


class CriticInterface:
    def __init__(self):
        pass

    @abstractmethod
    def provide_feedback(self, *args):
        pass

    @abstractmethod
    def calculate_grads(self, *args):
        pass

    @abstractmethod
    def update(self, grads: dict):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
