import flax.linen as nn

from src.models.autoencoder.encoder import Encoder
from src.models.helpers import linear_layer_init

import jax.numpy as jnp


class DreamerEncoder(nn.Module):
    features: int

    def setup(self) -> None:
        self.kernel = 4
        self.strides = 2
        self.layers = 6
        self.pixel_embedding = linear_layer_init(features=self.features // 4)
        self.encoder = Encoder(self.features, self.kernel, self.strides, self.layers, True)

    def __call__(self, image):
        embedded_image = self.pixel_embedding(image)
        encoded_image, _ = self.encoder(embedded_image)
        encoded_image = jnp.reshape(encoded_image, (encoded_image.shape[0], -1))
        return encoded_image
