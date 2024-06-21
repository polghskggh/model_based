import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.enviroment.shape import Shape
from src.models.agent.atarinn import AtariNN
from src.models.modelwrapper import ModelWrapper
from src.pod.storage import TransitionStorage, store
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key


class DQNStrategy(StrategyInterface):
    def __init__(self):
        self._q_network: ModelWrapper = ModelWrapper(AtariNN(Shape()[0], 1), "dqncritic")
        self._target_q_network: ModelWrapper = ModelWrapper(AtariNN(Shape()[0], 1), "dqncritic")
        self._action_space = Shape()[1]
        self._discount_factor: float = Args().args.discount_factor

        self._batch_size: int = Args().args.batch_size
        self._start_steps: int = Args().args.start_steps
        self._update_every: int = Args().args.update_every

        self._iteration: int = 0
        self._data_pos: int = 0
        self._storage_size: int = Args().args.storage_size
        self._parallel_agents: int = Args().args.num_agents

        self._epsilon: float = Args().args.epsilon
        self._target_update_period = Args().args.target_update_period
        self._storage = self._init_storage()

        self._update_counter = 0

    @staticmethod
    def _init_storage():
        size = Args().args.storage_size
        action_shape = (size, )
        observation_shape = action_shape + Shape()[0]
        return TransitionStorage(observations=jnp.zeros((observation_shape)), actions=jnp.zeros((action_shape)),
                                 rewards=jnp.zeros((action_shape)), next_observations=jnp.zeros((observation_shape)))

    def store_flattened(self, old_state, selected_action, reward, new_state):
        start_idx = self._data_pos % self._storage_size
        end_idx = (self._data_pos + self._parallel_agents) % self._storage_size

        if end_idx < start_idx:
            mid_point = self._storage_size - start_idx
            old_state, old_state_res = jnp.split(old_state, [mid_point])
            selected_action, selected_action_res = jnp.split(selected_action, [mid_point])
            reward, reward_res = jnp.split(reward, [mid_point])
            new_state, new_state_res = jnp.split(new_state, [mid_point])

            self._storage = store(self._storage, slice(0, end_idx),
                                  observations=old_state_res, actions=selected_action_res,
                                  rewards=reward_res, next_observations=new_state_res)
            end_idx = self._storage_size

        self._storage = store(self._storage, slice(start_idx, end_idx), observations=old_state,
                              actions=selected_action, rewards=reward, next_observations=new_state)
        self._data_pos += self._parallel_agents

    def timestep_callback(self, old_state: jnp.ndarray, selected_action: int, reward: float, new_state: jnp.ndarray, done: bool):
        self.store_flattened(old_state, selected_action, reward, new_state)

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
        for _ in range(num_epochs):
            states, actions, rewards, next_states = self.sample(batch_size)
            next_actions = vmap(self._greedy_action)(next_states)
            next_values = self._target_q_network.forward(next_states, next_actions).reshape(-1)
            td_targets: jax.Array = rewards + self._discount_factor * next_values
            td_targets = jnp.expand_dims(td_targets, 1)

            grads = self._q_network.train_step(td_targets, states, actions)
            self._q_network.apply_grads(grads)

        self._update_counter += 1
        if self._update_counter % self._target_update_period == 0:
            self._target_q_network.params = self._q_network.params

    def select_action(self, states: jax.Array, store_trajectories=True) -> jnp.ndarray:
        batch_size = states.shape[0]
        key = Key().key()
        probs = jr.uniform(key, (batch_size, ))

        actions = jnp.where(probs > self._epsilon, vmap(self._greedy_action)(states), self._random_policy(batch_size))
        return jnp.squeeze(actions)

    def _greedy_action(self, state: jax.Array) -> jax.Array:
        actions = jnp.arange(self._action_space)
        mapped_fun = vmap(self._q_network.forward, in_axes=(None, 0))
        values = mapped_fun(state, actions)
        values = jnp.squeeze(values)
        selected_action = jnp.argmax(values, axis=0)
        return selected_action

    @staticmethod
    def _random_policy(size: int) -> jax.Array:
        action = jr.randint(Key().key(), (size, ), 0, Shape()[1])[0]
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
