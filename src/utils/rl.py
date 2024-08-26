import distrax
import jax
from jax import vmap, jit
import jax.numpy as jnp
import flax.linen as nn

import jax.random as jr
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key


@jit
def tile_image(image: jax.Array) -> jax.Array:
    """
    Tile an RGB image to fit into 256 categories per pixel
    """
    return vmap(vmap(tile_pixel))(image)

@jit
def tile_pixel(pixel: jax.Array) -> int:
    """
    Tile an RGB pixel to fit into 256 categories
    """
    pixel /= 40
    category: int = pixel[0].astype(int) * 36 + pixel[1].astype(int) * 6 + pixel[2].astype(int)
    return category


@jit
def reverse_tile_image(image: jax.Array) -> jax.Array:
    """
    Reverse tiled image from 256 categories to RGB
    """

    def reverse_tile_pixel(pixel: jax.Array) -> jax.Array:
        red = pixel // 36
        green = (pixel % 36) // 6
        blue = pixel % 6
        rgb_pixel = jnp.array([red, green, blue])
        return rgb_pixel

    image_sampled = jnp.argmax(image, axis=-1)
    return vmap(vmap(reverse_tile_pixel))(image_sampled)


@jit
def generalized_advantage_estimation(values, rewards, discounts, lambda_):
    lambda_ = jnp.ones_like(discounts) * lambda_  # make lambda into vector.
    truncated_values = values[:-1]
    td_errors = rewards + discounts * values[1:] - truncated_values

    def fold_left(accumulator, rest):
        td_error, discount, lambda_ = rest
        accumulator = td_error + discount * lambda_ * accumulator
        return accumulator, accumulator

    _, advantages = jax.lax.scan(fold_left, jnp.zeros(td_errors.shape[1]), (td_errors, discounts, lambda_),
                                 reverse=True)
    return advantages, advantages + truncated_values


def process_output(output):
    output = jnp.squeeze(output)
    if output.shape[-1] == 1:
        return jnp.squeeze(output)
    else:
        if Args().args.sample_output:
            distribution = distrax.Categorical(output)
            return distribution.sample(seed=Key().key()).squeeze()
        else:
            return jnp.argmax(output, axis=-1).squeeze()


def zero_on_term(dones, values):
    dones = jnp.broadcast_to(jnp.expand_dims(dones, -1), values.shape)
    return jnp.where(dones, jnp.zeros_like(values), values)
