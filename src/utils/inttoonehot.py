from ctypes import Array

import jax.numpy as jnp
from jax import vmap


def image_to_onehot(image: Array[Array[int]]) -> Array[Array[Array[int]]]:
    """
    Convert an image to a one-hot image
    """
    return vmap(vmap(int_to_onehot))(image)


def int_to_onehot(category: int) -> Array[int]:
    """
    Convert an integer category to a one-hot vector
    """
    return jnp.eye(256)[category]
