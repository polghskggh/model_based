from ctypes import Array

from src.agent.actor.actorinterface import ActorInterface
from src.agent.critic import CriticInterface
from src.models.modelwrapper import ModelWrapper
from flax import linen as nn
import numpy as np

from src.models.trainer.critictrainer import CriticTrainer


class DDPGCritic(CriticInterface):
    def __init__(self, model: nn.Module, discount_factor: float, polyak: float = 0.995, action_dim: int = 4):
        super().__init__()
        self._model: ModelWrapper = ModelWrapper(model, "critic")
        self._target_model: ModelWrapper = ModelWrapper(model, "critic")

        self._discount_factor: float = discount_factor
        self._polyak: float = polyak
        self._action_dim: int = action_dim
        self._trainer = CriticTrainer(self._model.model)

    def calculate_grads(self, state: Array[float], action: Array[float], reward: float,
                        next_state: Array[float], next_action: Array[float]) -> Array[float]:
        observed_values: Array[float] = (
                reward + self._discount_factor * self._target_model.forward(next_state, next_action).reshape(-1))
        return self._model.train_step(observed_values, state, action)

    def update(self, grads: dict):
        self._model.apply_grads(grads)
        self._target_model.update_polyak(self._polyak, self._model)

    def provide_feedback(self, state: Array, action: Array) -> Array:
        return self._trainer.train_step(self._model.params, state, action)

    def update_common_head(self, actor: ActorInterface):
        actor.model.params["params"]["cnn"] = self._model.params["params"]["cnn"]

    def calculate_generalized_advantage_estimator(
            reward, value, done, gae_gamma, gae_lambda):
        # pylint: disable=g-doc-args
        """Generalized advantage estimator.

        Returns:
          GAE estimator. It will be one element shorter than the input; this is
          because to compute GAE for [0, ..., N-1] one needs V for [1, ..., N].
        """
        # pylint: enable=g-doc-args

        next_value = value[1:, :]
        next_not_done = 1 - tf.cast(done[1:, :], tf.float32)
        delta = (reward[:-1, :] + gae_gamma * next_value * next_not_done
                 - value[:-1, :])

        return_ = tf.reverse(tf.scan(
            lambda agg, cur: cur[0] + cur[1] * gae_gamma * gae_lambda * agg,
            [tf.reverse(delta, [0]), tf.reverse(next_not_done, [0])],
            tf.zeros_like(delta[0, :]),
            parallel_iterations=1), [0])
        return tf.check_numerics(return_, "return")