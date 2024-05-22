import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from src.agent.agentstrategy.agentstrategyfactory import agent_strategy_factory
from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.enviroment.shape import Shape
from src.singletons.hyperparameters import Args


class Agent:
    def __init__(self, strategy: str):
        super().__init__()
        self._old_state: jax.Array = None
        self._new_state: jax.Array = None

        self._selected_action: int = 0
        self._reward: float = 0
        self._done = False
        self._store_trajectories: bool = True
        self._strategy: StrategyInterface = agent_strategy_factory(strategy)

    def update_policy(self):
        if self.store_trajectories:
            self._strategy.timestep_callback(self._old_state, self._selected_action, self._reward, self._new_state,
                                             self._done)

    def select_action(self) -> jax.Array:
        self._selected_action = self._strategy.select_action(self._new_state, self.store_trajectories)
        return self._selected_action

    def receive_reward(self, reward: float):
        self._reward = reward

    def receive_state(self, state: jax.Array):
        self._old_state = self._new_state
        self._new_state = state

    def receive_done(self, done):
        self._done = done

    def save(self):
        self._strategy.save()

    def load(self):
        self._strategy.load()

    @property
    def store_trajectories(self):
        return self._store_trajectories

    @store_trajectories.setter
    def store_trajectories(self, new_value: bool):
        self._store_trajectories = new_value

    def last_transition(self):
        return self._old_state, self._selected_action, self._reward, self._new_state, self._done
