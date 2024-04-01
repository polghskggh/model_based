import flax.linen as nn


class BitPredictor(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        nn.LSTMCell(features=self.features)(x)

