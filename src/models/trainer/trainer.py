from abc import abstractmethod


class Trainer:
    def __init__(self):
        pass

    @abstractmethod
    def train_step(self, *data):
        pass
