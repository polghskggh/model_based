import jax
import jax.numpy as jnp
from rlax import one_hot

from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.agent.critic import DDPGCritic
from src.enviroment.shape import Shape
from src.models.actorcritic.atarinn import AtariNN
from src.pod import ReplayBuffer
from src.pod.hyperparameters import hyperparameters


class DQNStrategy(StrategyInterface):
    def __init__(self):
        self._batch_size: int = hyperparameters["ddpg"]['batch_size']
        self._batches_per_update: int = hyperparameters["ddpg"]['batches_per_update']
        self._q_network = DDPGCritic(AtariNN(*Shape(), 1))
        self._action_space = Shape()[1]

    def _batch_update(self, training_sample: list[jax.Array]):
        best_actions = self.select_action(training_sample[2])

        grads = self._q_network.calculate_grads(
            training_sample[0], training_sample[1], training_sample[3],
            training_sample[2], best_actions)

        self._q_network.update(grads)

    def update(self, buffer: ReplayBuffer):
        for _ in range(self._batches_per_update):
            self._batch_update(buffer.sample(self._batch_size))

    def select_action(self, state: jnp.ndarray) -> jnp.ndarray:
        is_batch: bool = len(state.shape) > 3

        batch_size = state.shape[0] if is_batch else 1
        for action in range(self._action_space):
            actions = jnp.zeros(batch_size) + action
            values = self._q_network._target_model.forward(state, one_hot(actions, self._action_space))

        selected_actions = one_hot(jnp.argmax(values, axis=-1), self._action_space)

        return selected_actions if is_batch else selected_actions[0]



