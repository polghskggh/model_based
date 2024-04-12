from ctypes import Array

from jax import vmap


def tile_image(image: Array[Array[float]]) -> Array[Array[int]]:
    """
    Tile an RGB image to fit into 256 categories per pixel
    """
    return vmap(vmap(tile_pixel))(image)


def tile_pixel(pixel: Array[float]) -> int:
    """
    Tile an RGB pixel to fit into 256 categories
    """
    pixel /= 40
    category: int = pixel[0].astype(int) * 36 + pixel[1].astype(int) * 6 + pixel[2].astype(int)
    return category
