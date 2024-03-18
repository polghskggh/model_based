import numpy as np

from src.agent.acstrategy import strategy, shapes
from src.agent.agentinterface import AgentInterface
from src.agent.actor.ddpgactor import DDPGActor
from src.pod import ReplayBuffer
from jax.tree_util import tree_map


class Agent(AgentInterface):
    def __init__(self, agent_type: str):
        super().__init__()
        self._old_state: np.ndarray[float] = np.array(shapes[agent_type][0], float)
        self._new_state: np.ndarray[float] = np.array(shapes[agent_type][0], float)
        self._selected_action: np.ndarray[float] = np.zeros(shapes[agent_type][1])
        self._reward: float = 0

        self._actor, self._critic = strategy[agent_type]

        self._replay_buffer: ReplayBuffer = ReplayBuffer(shapes[agent_type][0], shapes[agent_type][1])
        self._batch_size: int = 100

        self._batches_per_update: int = 5
        self._start_steps: int = 200

        self._update_after: int = 100
        self._update_every: int = 50
        self._iteration: int = self._update_every

    def _batch_update(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        training_sample: list[np.ndarray[float]] = self._replay_buffer.sample(self._batch_size)
        actor_actions = self._actor.calculate_actions(training_sample[2])
        critic_grads = self._critic.calculate_grads(
            training_sample[3], training_sample[0], training_sample[1],
            training_sample[2], actor_actions)

        action_grads = self._critic.provide_feedback(self._actor, training_sample[0])
        return critic_grads, action_grads

    def update_policy(self):
        self._replay_buffer.add_transition(self._old_state, self._selected_action, self._new_state, self._reward)

        # explore at start
        if self._start_steps != 0:
            self._start_steps -= 1
            return

        # only update after some number of time steps
        if self._iteration != self._update_every:
            self._iteration += 1
            return

        critic_grads_sum, actor_grads_sum = self._batch_update()
        for _ in range(self._batches_per_update - 1):
            critic_grads, actor_grads = self._batch_update()
            critic_grads_sum = Agent.sum_dicts(critic_grads_sum, critic_grads)
            actor_grads_sum = Agent.sum_dicts(actor_grads_sum, actor_grads)

        self._critic.update(tree_map(lambda x: x / self._batches_per_update, critic_grads_sum))
        self._actor.update(tree_map(lambda x: -x / self._batches_per_update, actor_grads_sum))
        self._iteration = 0

    def select_action(self) -> np.ndarray[float]:
        if self._start_steps != 0:
            self._start_steps -= 1
            return Agent._random_action()

        self._selected_action = self._actor.approximate_best_action(self._new_state)
        return self._selected_action

    def receive_reward(self, reward: float):
        self._reward = reward

    def receive_state(self, state: np.ndarray[float]):
        self._old_state = self._new_state
        self._new_state = state

    @staticmethod
    def sum_dicts(dict1, dict2):
        return tree_map(lambda x, y: x + y, dict1, dict2)

    @staticmethod
    def _random_action():
        return DDPGActor.softmax_to_onehot(np.random.rand(4))
