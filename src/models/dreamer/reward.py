from flax import linen as nn

from src.models.autoencoder.decoder import Decoder


class RewardModel(nn.Module):
    def setup(self):
        self.features = 256
        self.kernel = (4, 4)
        self.strides = (2, 2)
        self.layers = 6
        self.decoder = Decoder(self.features, self.kernel, self.strides, self.layers, self.deterministic)

    @nn.compact
    def __call__(self, x):
        x = self.decoder(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(1)(x)
        return x
