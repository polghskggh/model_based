from abc import abstractmethod


class Trainer:
    @abstractmethod
    def train_step(self, *data):
        pass
