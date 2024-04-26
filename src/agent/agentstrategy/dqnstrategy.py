import jax
import jax.numpy as jnp
import jax.random as jr
from rlax import one_hot


from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.agent.critic import DQNNetwork
from src.enviroment.shape import Shape
from src.models.actorcritic.atarinn import AtariNN
from src.pod.hyperparameters import hyperparameters
from src.pod.replaybuffer import ReplayBuffer


class DQNStrategy(StrategyInterface):

    def __init__(self):
        self._replay_buffer = ReplayBuffer(*Shape())
        self._batches_per_update: int = hyperparameters["ddpg"]['batches_per_update']
        self._q_network = DQNNetwork(AtariNN(*Shape(), 1))
        self._action_space = Shape()[1]

        self._batch_size: int = hyperparameters["dqn"]['batch_size']
        self._start_steps: int = hyperparameters["dqn"]["start_steps"]
        self._update_every: int = hyperparameters["dqn"]["update_every"]
        self._iteration: int = 0
        self._key = jr.PRNGKey(hyperparameters["rng"]["action"])

    def _batch_update(self, training_sample: list[jax.Array]):
        best_actions = self.action_policy(training_sample[2])

        grads = self._q_network.calculate_grads(
            training_sample[0], training_sample[1], training_sample[3],
            training_sample[2], best_actions)

        self._q_network.update(grads)

    def update(self, old_state: jnp.ndarray, selected_action: int, reward: float, new_state: jnp.ndarray, done: bool):
        self._replay_buffer.add_transition(old_state, selected_action, reward, new_state)

        # explore at start
        if self._start_steps != 0:
            self._start_steps -= 1
            return

        # only update after some number of time steps
        if self._iteration != self._update_every:
            self._iteration += 1
            return

        self._iteration = 0

        for _ in range(self._batches_per_update):
            self._batch_update(self._replay_buffer.sample(self._batch_size))

    def action_policy(self, state: jnp.ndarray) -> jnp.ndarray:
        is_batch: bool = len(state.shape) > 3

        batch_size = state.shape[0] if is_batch else 1
        for action in range(self._action_space):
            actions = jnp.zeros(batch_size) + action
            values = self._q_network._target_model.forward(state, one_hot(actions, self._action_space))

        selected_actions = one_hot(jnp.argmax(values, axis=-1), self._action_space)

        return selected_actions if is_batch else selected_actions[0]

    def __random_policy(self):
        self._key, subkey = jr.split(self._key)
        actions = jr.randint(subkey, (1, ), 0, Shape()[1])[0]
        return actions

    def save(self):
        self._q_network.save()

    def load(self):
        self._q_network.load()
