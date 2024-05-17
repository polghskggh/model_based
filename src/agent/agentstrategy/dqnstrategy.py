import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.enviroment.shape import Shape
from src.models.actorcritic.atarinn import AtariNN
from src.models.modelwrapper import ModelWrapper
from src.pod.storage import DQNStorage, store
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key


class DQNStrategy(StrategyInterface):
    def __init__(self):
        self._q_network: ModelWrapper = ModelWrapper(AtariNN(*Shape(), 1, False), "dqncritic")
        self._target_q_network: ModelWrapper = ModelWrapper(AtariNN(*Shape(), 1, True), "dqncritic")
        self._action_space = Shape()[1]
        self._discount_factor: float = Args().args.discount_factor

        self._batch_size: int = Args().args.batch_size
        self._start_steps: int = Args().args.start_steps
        self._update_every: int = Args().args.update_every
        self._iteration: int = 0
        self._data_pos: int = 0
        self._epsilon: float = Args().args.epsilon
        self._target_update_period = Args().args.target_update_period
        self._storage = self._init_storage()

    @staticmethod
    def _init_storage():
        size = Args().args.storage_size
        action_shape = (size, )
        observation_shape = action_shape + Shape()[0]
        return DQNStorage(observations=observation_shape, actions=action_shape, rewards=action_shape,
                          next_observations=observation_shape)


    def update(self, old_state: jnp.ndarray, selected_action: int, reward: float, new_state: jnp.ndarray, done: bool):
        self._storage = store(self._storage, self._data_pos, observations=old_state, actions=selected_action,
                              rewards=reward, next_observations=new_state)

        self._data_pos += 1

        # explore at start
        if self._start_steps != 0:
            self._start_steps -= 1
            return

        # only update after some number of time steps
        if self._iteration != self._update_every:
            self._iteration += 1
            return

        self._iteration = 0

        num_epochs, batch_size = Args().args.num_epochs, Args().args.batch_size
        states, actions, rewards, next_states = self.sample(num_epochs * batch_size)

        for start_idx in range(0, num_epochs * batch_size, batch_size):
            end_idx = start_idx + batch_size
            batch_slice = slice(start_idx, end_idx)

            next_actions = vmap(self._greedy_action)(next_states[batch_slice])
            next_values = self._target_q_network.forward(next_states[batch_slice], next_actions).reshape(-1)
            td_targets: jax.Array = rewards + self._discount_factor * next_values
            td_targets = jnp.expand_dims(td_targets, 1)

            grads = self._q_network.train_step(td_targets, states[batch_slice], actions[batch_slice])
            self._q_network.apply_grads(grads)

        self._target_q_network.params = self._q_network.params

    def select_action(self, states: jax.Array) -> jnp.ndarray:
        is_batch: bool = len(states.shape) > 3
        batch_size = states.shape[0] if is_batch else 1
        key = Key().key(1)
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

    def sample(self, n: int) -> tuple:
        filled_values = min(self._data_pos, Args().args.storage_size)
        idx = jr.choice(Key().key(1), filled_values, (n,), True)
        return (self._storage.observations[idx], self._storage.actions[idx],
                self._storage.rewards[idx], self._storage.next_observations[idx])
