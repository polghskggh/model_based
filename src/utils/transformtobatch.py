import jax.numpy as jnp


def transform_to_batch(data, batch_dim=None):
    """
    :param data: network input
    :param batch_dim: batch dimension

    :return: data with batch dimension
    """
    if batch_dim is None:
        return data

    if data.ndim < batch_dim:
        return jnp.expand_dims(data, 0)
