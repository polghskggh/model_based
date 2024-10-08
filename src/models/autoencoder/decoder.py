from flax import linen as nn
from jax import Array
import jax.numpy as jnp

from src.models.helpers import transpose_convolution_layer_init


# decoder
class Decoder(nn.Module):
    features: int = 256
    kernel: int = 4
    strides: int = 2
    layers: int = 6
    deterministic: bool = True
    dropout: float = 0.15
    normalization: bool = True

    def setup(self):
        self.shape_list = [(3, 3), (6, 6), (11, 11), (21, 21), (42, 42), (84, 84)]

    @nn.compact
    def __call__(self, x: Array, skip: list[Array] | None) -> Array:
        if skip is not None:
            skip.reverse()

        for layer_id in range(self.layers):
            features = self.scaled_features(layer_id) # first 2 layers more features

            x = nn.Dropout(rate=self.dropout, deterministic=self.deterministic)(x)
            if self.normalization:
                x = nn.LayerNorm()(x)
            x = transpose_convolution_layer_init(features=features, kernel_size=self.kernel, strides=self.strides,
                                                 padding="SAME")(x)
            x = nn.relu(x)
            x = self.scale_image(x, self.shape_list[layer_id])
            if skip is not None:
                x = nn.LayerNorm()(x + skip[layer_id])
        return x

    def scaled_features(self, layer_id: int):
        if layer_id < 4:
            return self.features
        if layer_id == 4:
            return self.features // 2
        if layer_id == 5:
            return self.features // 4

    @staticmethod
    def scale_image(x: Array, shape: tuple):
        return x[:, :shape[0], :shape[1]]
