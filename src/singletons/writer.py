import time

from singleton_decorator import singleton
from tensorboardX import SummaryWriter

from src.singletons.hyperparameters import Args
from src.singletons.step_traceker import StepTracker
from src.utils.save_name import save_name


@singleton
class Writer:
    _writer: SummaryWriter = None

    @property
    def writer(self) -> SummaryWriter:
        if self._writer is None:
            self._writer = self.__init_writer()
        return self._writer

    @staticmethod
    def __init_writer():
        writer = SummaryWriter(f'runs/{save_name()}')
        return writer


def log(aux):
    writer = Writer().writer
    for key, value in aux.items():
        writer.add_scalar(f'updateaux/{key}', value, global_step=int(StepTracker()))
