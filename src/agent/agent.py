import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from src.agent.agentstrategy.agentstrategyfactory import agent_strategy_factory
from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.enviroment.shape import Shape
from src.pod.hyperparameters import hyperparameters


class Agent:
    def __init__(self, agent_type: str):
        super().__init__()
        self._old_state: jax.Array = jnp.array(Shape.shape[0], float)
        self._new_state: jax.Array = jnp.array(Shape.shape[0], float)

        self._selected_action: int = 0
        self._reward: float = 0
        self._done = False

        self._strategy: StrategyInterface = agent_strategy_factory(agent_type)

    def update_policy(self):
        self._strategy.update(self._old_state, self._selected_action, self._reward, self._new_state, self._done)

    def select_action(self) -> int:
        self._selected_action = self._strategy.select_action(self._new_state)
        return self._selected_action

    def receive_reward(self, reward: float):
        self._reward = reward

    def receive_state(self, state: jax.Array):
        self._old_state = self._new_state
        self._new_state = state

    def receive_term(self, done: bool):
        self._done = done

    def run_parallel(self, parallel_agents: int):
        self._strategy.run_parallel(parallel_agents)

    def save(self):
        self._strategy.save()

    def load(self):
        self._strategy.load()

    def last_transition(self):
        return self._old_state, self._selected_action, self._reward, self._new_state
