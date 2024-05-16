import distrax
import jax
import rlax
from flax import linen as nn
import jax.numpy as jnp
from jax import random as jr, value_and_grad, jit, lax

from src.enviroment import Shape
from src.models.actorcritic.actoratari import ActorAtari
from src.models.modelwrapper import ModelWrapper
from src.singletons.hyperparameters import Args


class Actor:
    def __init__(self):
        train_model = ActorAtari(*Shape(), True)
        self._model: ModelWrapper = ModelWrapper(train_model, "actor",
                                                 train_model=train_model)
        self._new_states = None
        self.key = jr.PRNGKey(0)
        self._train_model = train_model
        self._clip_threshold = Args().args.clip_threshold
        self._regularization = Args().args.regularization
        self._rng = ModelWrapper.make_rng_keys()

    def policy(self, states: jax.Array) -> distrax.Categorical:
        logits = self._model.forward(states)
        prob = distrax.Categorical(logits)
        return prob

    def calculate_grads(self, states: jax.Array, advantage: jax.Array, action: jax.Array, old_log_odds: jax.Array):
        grad_fun = value_and_grad(Actor.ppo_grad, 1)
        loss, grads = grad_fun(self._train_model, self._model.params, states, advantage, action, old_log_odds,
                               self._clip_threshold, self._regularization, self._rng)
        return grads

    @staticmethod
    def ppo_grad(model, params: dict, states: jax.Array, advantage: jax.Array, action_index: jax.Array,
                 old_log_odds: jax.Array, epsilon: float, regularization: float, rng: dict):
        policy = jit(model.apply)(params, states, rngs=rng)
        prob = jnp.take_along_axis(policy, jnp.expand_dims(action_index, 1), axis=1)
        prob = jnp.squeeze(prob)

        old_log_odds = jnp.take_along_axis(old_log_odds, jnp.expand_dims(action_index, 1), axis=1)
        old_log_odds = jnp.squeeze(old_log_odds)
        log_ratio = prob - old_log_odds
        ratio = jnp.exp(prob - old_log_odds)
        approx_kl = ((ratio - 1) - log_ratio).mean()

        loss = rlax.clipped_surrogate_pg_loss(ratio, advantage, epsilon)

        regularization = jnp.ones_like(advantage) * regularization
        entropy_loss = rlax.entropy_loss(policy, regularization)

        return loss + entropy_loss

    def update(self, grads: dict):
        self._model.apply_grads(grads)

    @property
    def model(self):
        return self._model

    def save(self):
        self._model.save("actor")

    def load(self):
        self._model.load("actor")


