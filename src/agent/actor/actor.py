import flax.linen as nn
from jax import random as jr
import jax

from src.models.modelwrapper import ModelWrapper
from src.trainer.actortrainer import PPOActorTrainer


class Actor:
    def __init__(self, model: nn.Module, strategy: str):
        super().__init__()
        self._model: ModelWrapper = ModelWrapper(model, strategy)
        self._trainer = PPOActorTrainer(self._model.model) if strategy == "ppo" else DreamerActorTrainer(self._model.model)
        self._new_states = None
        self.key = jr.PRNGKey(0)

    def calculate_actions(self, states: jax.Array) -> jax.Array:
        return self._model.forward(states)

    def calculate_grads(self, states: jax.Array, advantage: jax.Array, action: jax.Array) -> dict:
        grads = self._trainer.train_step(self._model.params, states, advantage, action)
        return grads

    def update(self, grads: dict):
        self._model.apply_grads(grads)

    @property
    def model(self):
        return self._model

    def save(self):
        self._model.save("actor")

    def load(self):
        self._model.load("actor")


