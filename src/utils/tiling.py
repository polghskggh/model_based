
import jax
from jax import vmap
import jax.numpy as jnp

def tile_image(image: jax.Array) -> jax.Array:
    """
    Tile an RGB image to fit into 256 categories per pixel
    """
    return vmap(vmap(tile_pixel))(image)


def tile_pixel(pixel: jax.Array) -> int:
    """
    Tile an RGB pixel to fit into 256 categories
    """
    pixel /= 40
    category: int = pixel[0].astype(int) * 36 + pixel[1].astype(int) * 6 + pixel[2].astype(int)
    return category


def reverse_tile_image(image: jax.Array) -> jax.Array:
    """
    Reverse tiled image from 256 categories to RGB
    """
    image_sampled = jnp.argmax(image, axis=-1)
    return vmap(reverse_tile_pixel)(image_sampled)


def reverse_tile_pixel(pixel: jax.Array) -> jax.Array:
    """
    Reverse tiled pixel from 256 categories to RGB
    """
    red = pixel // 36
    green = (pixel % 36) // 6
    blue = pixel % 6
    rgb_pixel = jnp.array([red, green, blue])
    return rgb_pixel

