import jax.numpy as jnp
import optax
from flax import linen as nn

from src.models.strategy.modelstrategy import ModelStrategy
from src.pod.hyperparameters import hyperparameters
from src.resultwriter import ModelWriter
from src.resultwriter.modelwriter import writer_instances


class PPOActorStrategy(ModelStrategy):
    def __init__(self):
        super().__init__()

    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32), )

    def batch_dims(self) -> tuple:
        return (4, ), (2,)

    def init_writer(self) -> ModelWriter:
        writer_instances["actor"] = ModelWriter("actor", ["actor_loss"])
        return writer_instances["actor"]

    def init_optim(self, learning_rate: float):
        tx = optax.chain(
            optax.clip_by_global_norm(hyperparameters["max_grad_norm"]),
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=PPOActorStrategy.linear_schedule,
                eps=1e-5
            )
        )

    @staticmethod
    def linear_schedule(count):
        num_mini_batches = hyperparameters["ppo"]["trajectory_length"] * hyperparameters["ppo"]["number_of_trajectories"] // hyperparameters["ppo"]["batch_size"]

        frac = 1.0 - (count // num_mini_batches) / 1e-7

        return hyperparameters["ppo"]["actor_lr"] * frac
