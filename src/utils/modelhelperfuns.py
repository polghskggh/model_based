import jax.random as jr
import jax.numpy as jnp


def sample_normal(rng, mean, std):
    return jr.normal(rng, std.shape) * std + mean


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

    return data
