import flax.linen as nn

from src.models.helpers import convolution_layer_init


class MiddleNetwork(nn.Module):
    features: int
    kernel: int
    deterministic: bool = True
    dropout: float = 0.15

    @nn.compact
    def __call__(self, x):
        for layer in range(2):
            x = nn.Dropout(rate=self.dropout, deterministic=self.deterministic)(x)
            next_activation = convolution_layer_init(features=self.features, kernel_size=self.kernel,
                                                     strides=1, padding="SAME")(x)
            next_activation = nn.relu(next_activation)
            if layer == 0:
                x = next_activation
            else:
                x = nn.LayerNorm()(x + next_activation)
        return x
