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
        x = nn.ConvTranspose(features=9, kernel_size=(4, 4))(image) * jnp.argmax(actions)
        x = nn.ConvTranspose(features=9, kernel_size=(4, 4))(image) * jnp.argmax(actions)
        x = nn.ConvTranspose(features=9, kernel_size=(4, 4))(image) * jnp.argmax(actions)
        return x

    def unflatten(self, x: Array) -> Array:
        return x.reshape(self.output_dimensions) if x.ndim == 1 else x.reshape((x.shape[0], self.output_dimensions))
