import jax
import jax.numpy as jnp
import rlax
from jax import value_and_grad, vmap, lax, jit

from src.models.modelwrapper import ModelWrapper
from src.models.trainer.trainer import Trainer
from src.pod.hyperparameters import hyperparameters
from src.resultwriter.modelwriter import writer_instances


class PPOActorTrainer(Trainer):
    def __init__(self, model: ModelWrapper):
        super().__init__()
        self._model = model
        self._clip_threshold = hyperparameters["ppo"]["clip_threshold"]
        self._rng = ModelWrapper.make_rng_keys()

    def train_step(self, params: dict, states: jax.Array, advantage: jax.Array, action_index: jax.Array):
        batch_size = min(hyperparameters["ppo"]["batch_size"], states.shape[0] + states.shape[1])

        states = states.reshape(-1, batch_size, *states.shape[2:])
        advantage = advantage.reshape(-1, batch_size, *advantage.shape[2:])
        action_index = action_index.reshape(-1, batch_size, *action_index.shape[2:])

        grad_fun = value_and_grad(PPOActorTrainer.batch_ppo_grad, 1)
        loss, grads = grad_fun(self._model, params, states, advantage, action_index, self._clip_threshold,
                               self._rng)

        writer_instances["actor"].add_data(loss)
        return grads

    @staticmethod
    def ppo_grad(model, params: dict, states: jax.Array, advantage: jax.Array, action_index: int,
                 epsilon: float, rng: dict):
        policy = jit(model.apply)(params, states, rngs=rng)
        prob = jnp.take_along_axis(policy, action_index, axis=1)
        old_prob = lax.stop_gradient(prob)
        loss = rlax.clipped_surrogate_pg_loss(prob / old_prob, advantage, epsilon)
        return jnp.mean(loss)

    @staticmethod
    def batch_ppo_grad(model, params: dict, states: jax.Array, advantage: jax.Array, action_index: jax.Array,
                       epsilon: float, rng: dict):
        batch_loss = 0
        for state_b, advantage_b, action_index_b in zip(states, advantage, action_index):
            batch_loss += PPOActorTrainer.ppo_grad(model, params, state_b, advantage_b, action_index_b, epsilon, rng)

        batch_loss /= len(states)
        return batch_loss
