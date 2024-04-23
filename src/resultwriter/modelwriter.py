from enum import Enum

from src.resultwriter.abstractmodelwriter import AbstractModelWriter
from src.resultwriter.csvwriter import CsvWriter
from src.resultwriter.onlinedatatracker import OnlineDataTracker


class ModelWriter:
    def __init__(self, filename, tracked_values: list[str]):
        super().__init__()

        self._trackers = {value: OnlineDataTracker() for value in tracked_values}
        headers = ["episode"]

        for value in tracked_values:
            headers.append(value + "_mean")
            headers.append(value + "_var")

        self._tracked_values = tracked_values
        self._data = []
        self._csv_writer = CsvWriter(filename, headers)
        self._episode = 0

    def save_episode(self):
        self._data.append(self._prepare_episode_data())
        self._episode += 1

    def flush_buffer(self):
        self._csv_writer.store_data(self._data)
        self._data = []

    def _prepare_episode_data(self):
        episode_data = [self._episode]
        for var in self._tracked_values:
            mean, var = self._trackers[var].get_curr_mean_variance()
            episode_data.append(mean)
            episode_data.append(var)
        return episode_data

    # add new data
    def add_data(self, value: float, category: str = None):
        if category is None:
            category = self._tracked_values[0]
        self._trackers[category].update_aggr(value)

    @staticmethod
    def flush_all():
        for instance in writer_instances.values():
            instance.flush_buffer()

    @staticmethod
    def save_all():
        for instance in writer_instances.values():
            instance.save_episode()


writer_instances = {}
