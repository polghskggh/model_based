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
from src.singletons.step_traceker import StepTracker
from src.singletons.writer import Writer


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
        self._writer = Writer().writer

    def policy(self, states: jax.Array) -> distrax.Categorical:
        logits = self._model.forward(states)
        prob = distrax.Categorical(logits)
        return prob

    def calculate_grads(self, states: jax.Array, action: jax.Array, old_log_odds: jax.Array, advantage: jax.Array):
        grad_fun = value_and_grad(Actor.ppo_grad, 1, has_aux=True)
        (loss, aux), grads = grad_fun(self._train_model, self._model.params, states, advantage, action, old_log_odds,
                                      self._clip_threshold, self._regularization, self._rng)

        for key, value in aux.items():
            self._writer.add_scalar(f"losses/{key}", value, int(StepTracker()))

        return grads

    @staticmethod
    def ppo_grad(model, params: dict, states: jax.Array, advantage: jax.Array, action_index: jax.Array,
                 old_log_probs: jax.Array, epsilon: float, regularization: float, rng: dict):
        action_logits = jit(model.apply)(params, states, rngs=rng)
        policy = distrax.Categorical(action_logits)
        log_probs = policy.log_prob(action_index)

        log_ratio = log_probs - old_log_probs
        ratio = jnp.exp(log_ratio)

        approx_kl = ((ratio - 1) - log_ratio).mean()

        loss = rlax.clipped_surrogate_pg_loss(ratio, advantage, epsilon, False)
        entropy_loss = policy.entropy().mean()

        return (loss + regularization * entropy_loss,
                {"policy_loss": loss, "entropy_loss": entropy_loss, "kl_divergence": approx_kl})

    def update(self, grads: dict):
        self._model.apply_grads(grads)

    @property
    def model(self):
        return self._model

    def save(self):
        self._model.save()

    def load(self):
        self._model.load("actor")


