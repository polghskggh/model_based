import optax
from flax import linen as nn
from jax import numpy as jnp

from src.models.initalizer.helper import linear_schedule
from src.models.initalizer.modelstrategy import ModelStrategy
from src.models.lossfuns import cross_entropy_loss
from src.singletons.hyperparameters import Args


class AutoEncoderInitializer(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32),
                jnp.ones(1))

    def batch_dims(self) -> tuple:
        out_dims = (3, 1, 1) if Args().args.predict_dones else (3, 1)

        return (4, 1), out_dims

    def loss_fun(self):
        return cross_entropy_loss

    # def init_optim(self):
    #     returns optax.chain(
    #         optax.clip_by_global_norm(Args().args.max_grad_norm),
    #         optax.inject_hyperparams(optax.adafactor)(
    #             learning_rate=linear_schedule,
    #             eps=1e-5
    #         )
    #     )
