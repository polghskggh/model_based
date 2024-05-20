from abc import abstractmethod
from typing import Tuple

import flax.linen as nn
import optax

from src.models.initalizer.helper import linear_schedule
from src.models.lossfuns import mean_squared_error

from src.singletons.hyperparameters import Args


class ModelStrategy:
    @abstractmethod
    def init_params(self, model: nn.Module) -> tuple:
        pass

    def batch_dims(self) -> Tuple:
        return None, None

    def init_optim(self):
        return optax.chain(
            optax.clip_by_global_norm(Args().args.max_grad_norm),
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=linear_schedule,
                eps=1e-5
            )
        )

    def loss_fun(self):
        return mean_squared_error

