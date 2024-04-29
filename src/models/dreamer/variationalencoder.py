import flax.linen as nn

from src.models.autoencoder.encoder import Encoder


class VariationalEncoder(nn.Module):
    def setup(self):
        self.features = 256
        self.kernel = (4, 4)
        self.strides = (2, 2)
        self.layers = 6
        self.encoder = Encoder(self.features, self.kernel, self.strides, self.layers)

    @nn.compact
    def __call__(self, image):
        pass


