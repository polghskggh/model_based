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
        self._old_state: jax.Array = None
        self._new_state: jax.Array = None

        self._selected_action: int = 0
        self._reward: float = 0
        self._term = False
        self._trunc = False

        self._key = jr.PRNGKey(hyperparameters["rng"]["action"])
        self._strategy: StrategyInterface = agent_strategy_factory(agent_type)

    def update_policy(self):
        self._strategy.update(self._old_state, self._selected_action, self._reward,
                              self._new_state, self._term, self._trunc)

    def select_action(self) -> jax.Array:
        self._selected_action = self._strategy.select_action(self._new_state)
        return self._selected_action

    def receive_reward(self, reward: float):
        self._reward = reward

    def receive_state(self, state: jax.Array):
        self._old_state = self._new_state
        self._new_state = state

    def receive_term(self, term: bool, trunc: bool = False):
        self._term = term
        self._trunc = trunc

    def __sample_from_distribution(self, distribution: jax.Array) -> jax.Array:
        self._key, subkey = jr.split(self._key)
        return jr.choice(subkey, Shape()[1], p=distribution)

    def run_parallel(self, parallel_agents: int):
        self._strategy.run_parallel(parallel_agents)

    def save(self):
        self._strategy.save()

    def load(self):
        self._strategy.load()

    def last_transition(self):
        return self._old_state, self._selected_action, self._reward, self._new_state
