import jax
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


def sum_dicts(grads1, grads2):
    grads1_vals, tree = jax.tree_flatten(grads1)
    grads2_vals, tree = jax.tree_flatten(grads2)

    for g1, g2 in zip(grads1_vals, grads2_vals):
        g1 += g2

    return jax.tree_unflatten(tree, grads1_vals)

