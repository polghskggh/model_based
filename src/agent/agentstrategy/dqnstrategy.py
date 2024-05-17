import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.enviroment.shape import Shape
from src.models.actorcritic.atarinn import AtariNN
from src.models.modelwrapper import ModelWrapper
from src.pod.replaybuffer import ReplayBuffer
from src.pod.storage import DQNStorage
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key


class DQNStrategy(StrategyInterface):
    def __init__(self):
        self._batches_per_update: int = Args().args.batches_per_update
        self._q_network: ModelWrapper = ModelWrapper(AtariNN(*Shape(), 1, False), "dqncritic")
        self._target_q_network: ModelWrapper = ModelWrapper(AtariNN(*Shape(), 1, True), "dqncritic")

        self._action_space = Shape()[1]
        self._discount_factor: float = Args().args.discount_factor

        self._batch_size: int = Args().args.batch_size
        self._start_steps: int = Args().args.start_steps
        self._update_every: int = Args().args.update_every
        self._iteration: int = 0
        self._key = jr.PRNGKey(Key().key(1))
        self._epsilon: float = Args().args.epsilon
        self._target_update_period = Args().args.target_update_period
        self._storage = self._init_storage()

    @staticmethod
    def _init_storage():
        size = Args().args.storage_size
        return DQNStorage(observations=jnp.empty((size, Shape()[0])), actions=jnp.empty(size + 1), rewards=size),

    def _batch_update(self, training_sample: list[jax.Array]):
        states, actions, rewards, next_states = training_sample
        next_actions = vmap(self._greedy_action)(next_states)
        td_error: jax.Array = (
                rewards + self._discount_factor * self._target_q_network.forward(next_states, next_actions).reshape(-1))

        td_error = jnp.expand_dims(td_error, 1)

        grads = self._q_network.train_step(td_error, states, actions)
        return grads

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

        for b_idx in range(self._batches_per_update):
            grads = self._batch_update(self._replay_buffer.sample(self._batch_size))
            self._q_network.apply_grads(grads)

            if b_idx % self._target_update_period == 0 and b_idx != 0:
                self._target_q_network.params = self._q_network.params

    def select_action(self, states: jax.Array) -> jnp.ndarray:
        is_batch: bool = len(states.shape) > 3
        batch_size = states.shape[0] if is_batch else 1

        self._key, key = jr.split(self._key)
        probs = jr.uniform(key, (batch_size, ))

        def epsilon_greedy(p, state):
            return self._greedy_action(state) if p > self._epsilon else self._random_policy()

        action_selection_fun = epsilon_greedy
        if is_batch:
            action_selection_fun = vmap(action_selection_fun, in_axes=(0, 0))

        selected_actions = action_selection_fun(probs, states)
        return jnp.squeeze(selected_actions)

    def _greedy_action(self, state: jax.Array) -> jax.Array:
        actions = jnp.eye(self._action_space)
        mapped_fun = vmap(self._q_network.forward, in_axes=(None, 0))
        values = mapped_fun(state, actions)
        values = jnp.squeeze(values)
        selected_action = jnp.argmax(values, axis=0)
        return selected_action

    def _random_policy(self) -> jax.Array:
        self._key, subkey = jr.split(self._key)
        action = jr.randint(subkey, (1, ), 0, Shape()[1])[0]
        return action

    def save(self):
        self._q_network.save()
        self._target_q_network.save()

    def load(self):
        self._q_network.load("dqnnetwork")
        self._target_q_network.load("dqn_target_network")
