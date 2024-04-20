import jax.numpy as jnp
import jax.random as jr


class ReplayBuffer:
    def __init__(self, state_shape: tuple[int], action_shape: int):
        self._old_states = jnp.empty((0, *state_shape))
        self._actions = jnp.empty((0, action_shape))
        self._new_states = jnp.empty((0, *state_shape))
        self._rewards = jnp.empty(0)
        self._limit = 1000
        self._key = jr.PRNGKey(0)

    # put data into buffer
    def add_transition(self, old_state: jnp.ndarray, action: jnp.ndarray, reward: float,
                       new_state: jnp.ndarray):
        self._old_states = jnp.append(self._old_states, jnp.array([old_state]), axis=0)
        self._actions = jnp.append(self._actions, jnp.array([action]), axis=0)
        self._new_states = jnp.append(self._new_states, jnp.array([new_state]), axis=0)
        self._rewards = jnp.append(self._rewards, reward)
        self._check_limit()

    # sample n samples from buffer
    def sample(self, n: int) -> list[jnp.ndarray[float]]:
        self._key, subkey = jr.split(self._key)
        idx = jr.choice(subkey, self._rewards.shape[0], (n,), False)
        return [self._old_states[idx], self._actions[idx], self._new_states[idx], self._rewards[idx]]

    def _check_limit(self):
        current = self._rewards.shape[0]
        if current > self._limit:
            amount = current - self._limit
            self._old_states = self._old_states[amount:]
            self._actions = self._actions[amount:]
            self._new_states = self._new_states[amount:]
            self._rewards = self._rewards[amount:]
