import time

from singleton_decorator import singleton
from tensorboardX import SummaryWriter

from src.singletons.hyperparameters import Args
from src.utils.save_name import save_name


@singleton
class Writer:
    _writer: SummaryWriter = None

    @property
    def writer(self):
        if self._writer is None:
            self._writer = self.__init_writer()
        return self._writer

    @staticmethod
    def __init_writer():
        writer = SummaryWriter(f'runs/{save_name()}')
        return writer
