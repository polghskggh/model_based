import jax
import jax.numpy as jnp
import jax.random as jr
from rlax import one_hot

from src.agent.agentinterface import AgentInterface
from src.agent.agentstrategy.agentstrategyfactory import agent_strategy_factory
from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.enviroment.shape import Shape
from src.pod.hyperparameters import hyperparameters
from src.pod.replaybuffer import ReplayBuffer


class Agent(AgentInterface):
    def __init__(self, agent_type: str):
        super().__init__()
        self._old_state: jax.Array = jnp.array(Shape.shape[0], float)
        self._new_state: jax.Array = jnp.array(Shape.shape[0], float)

        self._followed_policy: jax.Array = jnp.zeros(Shape.shape[1])
        self._selected_action: int = 0

        self._reward: float = 0

        self._key = jr.PRNGKey(hyperparameters["rng"]["action"])

        self._strategy: StrategyInterface = agent_strategy_factory(agent_type)
        self._replay_buffer = ReplayBuffer(*Shape())

        self._start_steps: int = hyperparameters["agent"]["start_steps"]

        self._update_every: int = hyperparameters["agent"]["update_every"]
        self._iteration: int = self._update_every

    def update_policy(self):
        self._replay_buffer.add_transition(self._old_state, self._selected_action, self._reward,
                                           self._new_state)

        # explore at start
        if self._start_steps != 0:
            self._start_steps -= 1
            return

        # only update after some number of time steps
        if self._iteration != self._update_every:
            self._iteration += 1
            return

        self._strategy.update(self._replay_buffer)
        self._iteration = 0

    def select_action(self) -> jax.Array:
        if self._start_steps != 0:
            self._selected_action = self.__random_policy()
        else:
            followed_policy = self._strategy.action_policy(self._new_state)
            self._selected_action = self.__sample_from_distribution(followed_policy)

        return self._selected_action

    def receive_reward(self, reward: float):
        self._reward = reward

    def receive_state(self, state: jax.Array):
        self._old_state = self._new_state
        self._new_state = state

    def __sample_from_distribution(self, distribution: jax.Array) -> jax.Array:
        self._key, subkey = jr.split(self._key)
        return jr.choice(subkey, Shape()[1], p=distribution)

    def __random_policy(self):
        self._key, subkey = jr.split(self._key)
        actions = jr.randint(subkey, (1, ), 0, Shape()[1])[0]
        return actions

    def save(self):
        self._strategy.save()

    def load(self):
        self._strategy.load()

    @property
    def replay_buffer(self):
        return self._replay_buffer
