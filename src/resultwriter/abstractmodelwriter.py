from abc import abstractmethod


class AbstractModelWriter:
    def __init__(self):
        pass

    @abstractmethod
    def save_episode(self):
        pass

    @abstractmethod
    def add_data(self, loss: float):
        pass

    @abstractmethod
    def flush_buffer(self):
        pass
