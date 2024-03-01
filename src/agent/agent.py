import numpy as np

from src.agent.acstrategy import strategy
from src.agent.agentinterface import AgentInterface
from src.pod import ReplayBuffer


class Agent(AgentInterface):
    def __init__(self):
        super().__init__()
        self._old_state: np.ndarray[float] = np.array(24, float)
        self._new_state: np.ndarray[float] = np.array(24, float)
        self._selected_action: np.ndarray[float] = np.zeros(4)
        self._reward: float = 0

        self._actor, self._critic = strategy["ddpg"]

        self._replay_buffer: ReplayBuffer = ReplayBuffer()
        self._iteration: int = 0
        self._batch_size: int = 100
        self._forget_rate: float = 0.001

        self._batches_per_update: int = 5
        self._update_after: int = 200
        self._update_every: int = 50

    def _batch_update(self):
        training_sample: list[np.ndarray[float]] = self._replay_buffer.sample(self._batch_size)
        self._critic.update_model(
            training_sample[3], training_sample[0], training_sample[1],
            training_sample[2], self._actor.calculate_actions(training_sample[2]),
        )
        self._critic.provide_feedback(training_sample[0])

    def update_policy(self):
        self._replay_buffer.add_transition(self._old_state, self._selected_action, self._new_state, self._reward)

        # get experience before updating
        if self._update_after != 0:
            self._update_after -= 1
            return

        # only update after some number of time steps
        if self._iteration != self._update_every:
            self._iteration += 1
            return

        for _ in range(self._batches_per_update):
            self._batch_update()
            self._replay_buffer.forget(self._forget_rate)  # remove experiences from replay buffer
            self._iteration = 0

    def select_action(self) -> np.ndarray[float]:
        self._selected_action = self._actor.approximate_best_action(self._new_state)
        return self._selected_action

    def receive_reward(self, reward: float):
        self._reward = reward

    def receive_state(self, state: np.ndarray[float]):
        self._old_state = self._new_state
        self._new_state = state
