import numpy as np


class ReplayBuffer:
    def __init__(self, state_shape: tuple[int], action_shape: int):
        self._old_states = np.empty((0, *state_shape))
        self._actions = np.empty((0, action_shape))
        self._new_states = np.empty((0, *state_shape))
        self._rewards = np.empty(0)

    # put data into buffer
    def add_transition(self, old_state: np.ndarray[float], action: np.ndarray[float],
                       new_state: np.ndarray[float], reward: float):
        self._old_states = np.append(self._old_states, np.array([old_state]), axis=0)
        self._actions = np.append(self._actions, np.array([action]), axis=0)
        self._new_states = np.append(self._new_states, np.array([new_state]), axis=0)
        self._rewards = np.append(self._rewards, reward)

    # sample n samples from buffer
    def sample(self, n: int) -> list[np.ndarray[float]]:
        idx = np.random.choice(self._rewards.shape[0], (n,), False)
        return [self._old_states[idx], self._actions[idx], self._new_states[idx], self._rewards[idx]]

    # remove oldest entries
    def forget(self, factor: float):
        amount = int(factor * self._rewards.shape[0])
        self._old_states = self._old_states[amount:]
        self._actions = self._actions[amount:]
        self._new_states = self._new_states[amount:]
        self._rewards = self._rewards[amount:]
