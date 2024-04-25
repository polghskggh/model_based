import jax.numpy as jnp
import jax.random as jr
import jax
from jax import lax, vmap, jit


class ReplayBuffer:
    def __init__(self, state_shape: tuple[int], action_shape: int, limit: int = 1000):
        self._old_states = jnp.empty((0, *state_shape))
        self._actions = jnp.empty(0, dtype=jnp.int32)
        self._rewards = jnp.empty(0)
        self._new_states = jnp.empty((0, *state_shape))
        self._limit = limit
        self._key = jr.PRNGKey(0)

    # put data into buffer
    def add_transition(self, old_state: jax.Array, action: int, reward: float,
                       new_state: jax.Array):
        self._old_states = jnp.append(self._old_states, jnp.array([old_state]), axis=0)
        self._actions = jnp.append(self._actions, action)
        self._new_states = jnp.append(self._new_states, jnp.array([new_state]), axis=0)
        self._rewards = jnp.append(self._rewards, reward)
        self._check_limit()

    # sample n samples from buffer
    def sample(self, n: int, trajectory_length: int = 1) -> list[jax.Array]:
        self._key, subkey = jr.split(self._key)
        idx = jr.choice(subkey, self._rewards.shape[0] - (trajectory_length - 1), (n,), False)
        slice = ReplayBuffer.curried_slice_fun(trajectory_length)

        return [slice(self._old_states, idx),
                slice(self._actions, idx),
                slice(self._rewards, idx),
                slice(self._new_states, idx)]

    @staticmethod
    def curried_slice_fun(offset: int):
        def slice_fun(x: jax.Array, index: int) -> jax.Array:
            return lax.dynamic_slice_in_dim(x, index, offset)

        return vmap(jit(slice_fun), in_axes=(None, 0))

    def _check_limit(self):
        current = self._rewards.shape[0]
        if current > self._limit:
            amount = current - self._limit
            self._old_states = self._old_states[amount:]
            self._actions = self._actions[amount:]
            self._new_states = self._new_states[amount:]
            self._rewards = self._rewards[amount:]

    def data(self):
        return [self._old_states, self._actions, self._new_states, self._rewards]

    def __getitem__(self, item):
        return self._old_states[item], self._actions[item], self._new_states[item], self._rewards[item]