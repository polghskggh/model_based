import jax
import rlax
from flax import linen as nn
import jax.numpy as jnp
from jax import random as jr, value_and_grad, jit, lax

from src.enviroment import Shape
from src.models.actorcritic.actoratari import ActorAtari
from src.models.modelwrapper import ModelWrapper
from src.pod.hyperparameters import hyperparameters
from src.resultwriter.modelwriter import writer_instances


class Actor:
    def __init__(self):
        train_model = ActorAtari(*Shape(), True)
        self._model: ModelWrapper = ModelWrapper(train_model, "actor",
                                                 train_model=train_model)
        self._new_states = None
        self.key = jr.PRNGKey(0)
        self._train_model = train_model
        self._clip_threshold = hyperparameters["ppo"]["clip_threshold"]
        self._regularization = hyperparameters["ppo"]["regularization"]
        self._rng = ModelWrapper.make_rng_keys()

    def policy(self, states: jax.Array) -> jax.Array:
        logits = self._model.forward(states)
        prob = nn.softmax(logits)
        return prob, logits

    def calculate_grads(self, states: jax.Array, advantage: jax.Array, action: jax.Array, old_log_odds: jax.Array):
        grad_fun = value_and_grad(Actor.ppo_grad, 1)
        loss, grads = grad_fun(self._train_model, self._model.params, states, advantage, action, old_log_odds,
                               self._clip_threshold, self._regularization, self._rng)
        writer_instances["actor"].add_data(loss)
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

        Actor.debug(prob, action_index, advantage)

        regularization = jnp.ones_like(advantage) * regularization
        entropy_loss = rlax.entropy_loss(policy, regularization)

        jax.debug.print("entropy: {entropy}", entropy=entropy_loss)
        jax.debug.print("kl_term {kl_term}, ratio {ratio}", kl_term=approx_kl, ratio=ratio)
        jax.debug.print("log_probs {prob}, old_log_probs {old}", prob=prob, old=old_log_odds)
        return loss + entropy_loss

    @staticmethod
    def debug(prob, action_index, advantage):
        mean_p = [0, 0, 0, 0]
        mean_adv = [0, 0, 0, 0]
        counts = [0, 0, 0, 0]
        for p, a, adv in zip(prob, action_index, advantage):
            counts[a] += 1
            mean_p[a] += p
            mean_adv[a] += adv

        for i in range(4):
            mean_p[i] /= max(counts[i], 1)
            mean_adv[i] /= max(counts[i], 1)
            jax.debug.print("probs: {prob}, action: {index}, advantage: {adv}", prob=mean_p[i], index=i,
                            adv=mean_adv[i])

    def update(self, grads: dict):
        self._model.apply_grads(grads)

    @property
    def model(self):
        return self._model

    def save(self):
        self._model.save("actor")

    def load(self):
        self._model.load("actor")


