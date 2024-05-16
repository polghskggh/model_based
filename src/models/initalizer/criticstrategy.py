import jax.numpy as jnp
from flax import linen as nn

from src.models.initalizer.modelstrategy import ModelStrategy
from src.resultwriter.modelwriter import writer_instances, ModelWriter


class DQNCriticStrategy(ModelStrategy):
    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32),
                jnp.ones(model.second_input, dtype=jnp.float32))

    def batch_dims(self) -> tuple:
        return (4, 2), (2, )

    def init_writer(self):
        writer_instances["critic"] = ModelWriter("critic", ["critic_loss"])
        return writer_instances["critic"]


class CriticInitializer(ModelStrategy):
    def init_params(self, model: nn.Module) -> tuple:
        return (jnp.ones(model.input_dimensions, dtype=jnp.float32), )

    def batch_dims(self) -> tuple:
        return (4, ), (2, )

    def init_writer(self):
        writer_instances["critic"] = ModelWriter("critic", ["critic_loss"])
        return writer_instances["critic"]
