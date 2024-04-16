from ctypes import Array

import flax.linen as nn

from src.models.atari.autoencoder.autoencoder import AutoEncoder
from src.models.atari.inference.convolutionalinference import ConvolutionalInference


class TrainStochasticAutoencoder(nn.Module):
    input_dimensions: tuple
    second_input: int
    third_input: tuple

    def setup(self):
        self.autoencoder = AutoEncoder(self.input_dimensions, self.second_input)
        self.inference = ConvolutionalInference(self.input_dimensions, self.second_input, self.third_input, True)

    @nn.compact
    def __call__(self, stack: Array, actions: Array, next_frame: Array):
        bit_prediction = self.inference(stack, actions, next_frame)
        pixels = self.autoencoder(stack, actions, bit_prediction)
        return pixels