from flax import linen as nn
from jax import Array


# encoder
class Encoder(nn.Module):
    features: int
    kernel: tuple
    strides: tuple
    layers: int
    deterministic: bool
    dropout: float = 0.15

    @nn.compact
    def __call__(self, x: Array) -> tuple:
        skip = []
        for layer_id in range(self.layers):
            skip.append(x)
            features = self.scaled_features(layer_id)
            x = nn.Dropout(rate=self.dropout, deterministic=self.deterministic)(x)
            x = nn.LayerNorm()(x)
            x = nn.Conv(features=features, kernel_size=self.kernel, strides=self.strides)(x)
            x = nn.relu(x)
        return x, skip

    def scaled_features(self, layer_id: int):
        if layer_id >= 1:
            return self.features
        if layer_id == 0:
            return self.features // 2
