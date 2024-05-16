import time

from singleton_decorator import singleton
from tensorboardX import SummaryWriter

from src.singletons.hyperparameters import Args


@singleton
class Writer:
    _writer: SummaryWriter = None

    @property
    def writer(self):
        if self._writer is None:
            self._writer = Writer.__init_writer()
        return self._writer

    @staticmethod
    def __init_writer():
        args = Args()
        run_name = f"{args.seed}_{time.asctime(time.localtime(time.time())).replace('  ', ' ').replace(' ', '_')}"
        writer = SummaryWriter(f'runs/{args.env}/{args.algorihtm}/{args.env}/{run_name}')
        return writer
