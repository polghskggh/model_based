from src.resultwriter.abstractmodelwriter import AbstractModelWriter
from src.resultwriter.csvwriter import CsvWriter
from src.resultwriter.onlinedatatracker import OnlineDataTracker


class ModelWriter(AbstractModelWriter):
    def __init__(self, filename, loss_name: str):
        super().__init__()
        headers = ["episode", loss_name, "variance"]
        self.csv_writer = CsvWriter(filename, headers)
        self.tracker = OnlineDataTracker()
        self.losses_mean = []
        self.losses_var = []
        # flush data into file

    def save_episode(self):
        mean, var = self.tracker.get_curr_mean_variance()
        self.losses_mean.append(mean)
        self.losses_var.append(var)
        if len(self.losses_mean) == 1000:
            self.flush_buffer()

    def flush_buffer(self):
        data = self._prepare_data()
        self.csv_writer.store_data(data)
        self.losses_var = []
        self.losses_mean = []

    def _prepare_data(self):
        data = [[episode, mean, var] for episode, (mean, var) in enumerate(zip(self.losses_mean, self.losses_var))]
        return data

    # add new data
    def add_data(self, loss: float):
        self.tracker.update_aggr(loss)

    @staticmethod
    def flush_all():
        for instance in writer_instances.values():
            instance.flush_buffer()


writer_instances = {"actor": ModelWriter("actor", "actor_loss"),
                    "critic": ModelWriter("critic", "critic_loss"),
                    "reward": ModelWriter("reward", "reward"),
                    "autoencoder": ModelWriter("autoencoder", "image_loss"),
                    "mock": AbstractModelWriter()}
