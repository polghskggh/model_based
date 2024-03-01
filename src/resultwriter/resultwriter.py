import csv

from src.resultwriter.csvwriter import CsvWriter


class ResultWriter:
    def __init__(self, filename: str, headers: list[str]):
        prepend: list[str] = ["episode", "timestamp"]
        self.csv_writer = CsvWriter(filename, prepend + headers)

        self.episode = 0
        self.timestamp = 0
        self.observations = []
        self.actions = []
        self.rewards = []

    # flush data into file
    def save_episode(self):
        data = self._prepare_data()
        self.csv_writer.store_data(data)
        self.new_episode()

    def _prepare_data(self):
        data = [[self.episode] + [timestamp] + observations + actions + [reward]
                for timestamp, observations, actions, reward
                in zip(range(self.timestamp + 1), self.observations, self.actions, self.rewards)]
        return data

    # add new data
    def add_data(self, observation, action, reward: float):
        self.timestamp += 1
        self.observations.append(list(observation))
        self.actions.append(list(action))
        self.rewards.append(reward)

    # reset data, increment episode
    def new_episode(self):
        self.timestamp = 0
        self.episode += 1

        self.observations = []
        self.actions = []
        self.rewards = []
