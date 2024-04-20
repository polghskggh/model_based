import jax
import jax.numpy as jnp
import jax.random as jr
from rlax import one_hot

from src.agent.agentinterface import AgentInterface
from src.agent.agentstrategy.agentstrategyfactory import agent_strategy_factory
from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.agent.trajectory.trajectoryfactory import trajectory_factory
from src.enviroment.shape import Shape
from src.pod.hyperparameters import hyperparameters


class Agent(AgentInterface):
    def __init__(self, agent_type: str):
        super().__init__()
        self._old_state: jax.Array = jnp.array(Shape.shape[0], float)
        self._new_state: jax.Array = jnp.array(Shape.shape[0], float)
        self._selected_action: jax.Array = jnp.zeros(Shape.shape[1])
        self._reward: float = 0
        self._key = jr.PRNGKey(hyperparameters["rng"]["action"])

        self._strategy: StrategyInterface = agent_strategy_factory(agent_type)
        self._trajectory_storage = trajectory_factory(agent_type)

        self._batch_size: int = 100

        self._batches_per_update: int = 5
        self._start_steps: int = 200

        self._update_after: int = 100
        self._update_every: int = 50
        self._iteration: int = self._update_every

    def update_policy(self):
        self._trajectory_storage.add_transition(self._old_state, self._selected_action, self._reward, self._new_state)

        # explore at start
        if self._start_steps != 0:
            self._start_steps -= 1
            return

        # only update after some number of time steps
        if self._iteration != self._update_every:
            self._iteration += 1
            return

        self._strategy.update(self._trajectory_storage)
        self._iteration = 0

    def select_action(self) -> jax.Array:
        if self._start_steps != 0:
            self._start_steps -= 1
            self._selected_action = self._random_action()
        else:
            self._selected_action = self._strategy.select_action(self._new_state)

        return self._selected_action

    def receive_reward(self, reward: float):
        self._reward = reward

    def receive_state(self, state: jax.Array):
        self._old_state = self._new_state
        self._new_state = state

    def _random_action(self):
        self._key, subkey = jr.split(self._key)
        return one_hot(jr.randint(subkey, (1, ), 0, Shape()[1]), Shape()[1])[0]
