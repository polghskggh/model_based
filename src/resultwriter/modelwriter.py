import csv

from src.resultwriter.csvwriter import CsvWriter


class ModelWriter:
    def __init__(self, filename, loss_name: str):
        headers = ["episode", loss_name]
        self.csv_writer = CsvWriter(filename, headers)
        self.episode = 0
        self.losses = []
        # flush data into file

    def save_episode(self):
        data = self._prepare_data()
        self.csv_writer.store_data(data)
        self.losses = []

    def _prepare_data(self):
        data = [[self.episode] + [loss] for loss in self.losses]
        return data

    # add new data
    def add_data(self, loss: float):
        self.losses.append(loss)
