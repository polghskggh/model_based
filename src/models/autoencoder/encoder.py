import jax
from flax import linen as nn
from jax import Array
import jax.numpy as jnp

from src.models.helpers import convolution_layer_init


# encoder
class Encoder(nn.Module):
    features: int = 256
    kernel: tuple = 4
    strides: tuple = 2
    layers: int = 6
    deterministic: bool = True
    dropout: float = 0.15

    @nn.compact
    def __call__(self, x: Array) -> tuple:
        skip = []
        for layer_id in range(5):
            skip.append(x)
            features = self.scaled_features(layer_id)
            x = nn.Dropout(rate=self.dropout, deterministic=self.deterministic)(x)
            x = nn.LayerNorm()(x)
            strides = self.strides if layer_id < self.layers - 3 else 1
            print("encoder strides", strides)
            x = convolution_layer_init(features=features, kernel_size=self.kernel, strides=strides)(x)
            print("encoder", x.shape)
            x = nn.relu(x)
        return x, skip

    def scaled_features(self, layer_id: int):
        if layer_id >= 3:
            return self.features
        if layer_id == 2:
            return self.features // 2
        if layer_id == 1:
            return self.features // 4
        if layer_id == 0:
            return self.features // 8
