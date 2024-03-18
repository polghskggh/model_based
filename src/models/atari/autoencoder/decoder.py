from flax import linen as nn
from jax import Array
import jax.numpy as jnp


# decoder
class Decoder(nn.Module):
    input_dimensions: tuple
    second_input: int
    output_dimensions: tuple

    @nn.compact
    def __call__(self, image, actions) -> Array:
        x = nn.ConvTranspose(features=9, kernel_size=(4, 4), strides=(4, 4))(image)
        x = nn.ConvTranspose(features=9, kernel_size=(4, 4), strides=(4, 4))(x) * jnp.argmax(actions)
        return x
