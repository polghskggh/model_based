import jax.numpy as jnp
import jax.random as jr

from src.agent.actor.ddpgactor import DDPGActor
from src.agent.agentinterface import AgentInterface
from src.agent.agentstrategy.agentstrategyfactory import agent_strategy_factory
from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.agent.critic import DDPGCritic
from src.agent.trajectory.trajectoryfactory import trajectory_factory
from src.enviroment.shape import Shape
from src.pod.hyperparameters import hyperparameters
from src.utils.inttoonehot import softmax_to_onehot


class Agent(AgentInterface):
    def __init__(self, agent_type: str):
        super().__init__()
        self._old_state: jnp.ndarray = jnp.array(Shape.shape[0], float)
        self._new_state: jnp.ndarray = jnp.array(Shape.shape[0], float)
        self._selected_action: jnp.ndarray = jnp.zeros(Shape.shape[1])
        self._reward: float = 0
        self._key = jr.PRNGKey(hyperparameters["seed"])

        self._strategy: StrategyInterface = agent_strategy_factory(agent_type)
        self._trajectory_storage = trajectory_factory(agent_type)

        self._actor, self._critic = DDPGActor(*Shape.shape), DDPGCritic(*Shape.shape)

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

    def select_action(self) -> jnp.ndarray:
        if self._start_steps != 0:
            self._start_steps -= 1
            self._selected_action = Agent._random_action()
        else:
            self._selected_action = self._strategy.select_action(self._new_state)

        return self._selected_action

    def receive_reward(self, reward: float):
        self._reward = reward

    def receive_state(self, state: jnp.ndarray):
        self._old_state = self._new_state
        self._new_state = state

    @staticmethod
    def _random_action():
        return softmax_to_onehot(jr.rand(4))
