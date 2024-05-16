import jax.numpy as jnp
import optax
from flax import linen as nn

from src.models.initalizer.modelstrategy import ModelStrategy
from src.singletons.hyperparameters import Args


class ActorInitializer(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32), )

    def batch_dims(self) -> tuple:
        return (4, ), (2,)

    def init_optim(self, learning_rate: float):
        return optax.chain(
            optax.clip_by_global_norm(Args().args.max_grad_norm["max_grad_norm"]),
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=ActorInitializer.linear_schedule,
                eps=1e-5
            )
        )


    @staticmethod
    def linear_schedule(count):
        num_mini_batches = hyperparameters["ppo"]["trajectory_length"] * hyperparameters["ppo"]["number_of_trajectories"] // hyperparameters["ppo"]["batch_size"]

        frac = 1.0 - (count // num_mini_batches) / 1e-7

        return hyperparameters["ppo"]["actor_lr"] * frac
