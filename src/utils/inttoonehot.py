import jax.numpy as jnp
from jax.nn import one_hot


def softmax_to_onehot(actions: jnp.ndarray) -> jnp.ndarray:
    return one_hot(jnp.argmax(actions, axis=1), 4)
