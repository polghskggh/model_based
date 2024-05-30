from flax import linen as nn
from jax import Array
import jax.numpy as jnp


# decoder
class Decoder(nn.Module):
    features: int = 256
    kernel: tuple = (4, 4)
    strides: tuple = (2, 2)
    layers: int = 6
    deterministic: bool = True
    dropout: float = 0.15

    def setup(self):
        self.shape_list = [(4, 3), (7, 5), (14, 10), (27, 20), (53, 40), (105, 80)]

    @nn.compact
    def __call__(self, x: Array, skip: list[Array] | None) -> Array:
        if skip is not None:
            skip.reverse()

        for layer_id in range(self.layers):
            features = self.scaled_features(layer_id) # first 2 layers more features
            x = nn.Dropout(rate=self.dropout, deterministic=self.deterministic)(x)
            x = nn.LayerNorm()(x)
            x = nn.ConvTranspose(features=features, kernel_size=self.kernel, strides=self.strides)(x)
            x = nn.relu(x)
            x = self.scale_image(x, layer_id)
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

    def scale_image(self, x: Array, layer_id: int):
        if x.ndim == 4:
            return x[:, :self.shape_list[layer_id][0], :self.shape_list[layer_id][1]]

        return x[:self.shape_list[layer_id][0], :self.shape_list[layer_id][1]]
