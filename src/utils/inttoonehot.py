from ctypes import Array

import jax.numpy as jnp
from jax import vmap
from jax.nn import one_hot


def image_to_onehot(image: Array[Array[int]]) -> Array[Array[Array[int]]]:
    """
    Convert an image to a one-hot image
    """
    return vmap(one_hot, (0, None))(image, 256)
