import flax.linen as nn
from jax import Array

from src.models.atari.autoencoder.pixelembedding import PixelEmbedding


class ConvolutionalInference(nn.Module):
    input_dimensions: tuple
    second_input: int

    def setup(self):
        self.pixel_embedding = PixelEmbedding(64)
        self.features = 256
        self.kernel = (8, 8)
        self.strides = (4, 4)
        self.layers = 6


    @nn.compact
    def __call__(self, image: Array):
        embedded_image = self.pixel_embedding(image)
        conv = nn.Conv(embedded_image, kernel_size=self.kernel, strides=self.strides)(embedded_image)
        conv2 = nn.Conv(conv, kernel_size=self.kernel, strides=self.strides)(conv)

        conv = nn.MultiHeadDotProductAttention(conv)
        conv2 = nn.MultiHeadDotProductAttention(conv2)

        return conv2

