from ctypes import Array

import flax.linen as nn

from src.models.atari.autoencoder.autoencoder import AutoEncoder
from src.models.atari.inference.bitpredictior import BitPredictor


class StochasticAutoencoder(nn.Module):
    input_dimensions: tuple
    second_input: int
    bits: int = 128

    def setup(self):
        self.autoencoder = AutoEncoder(self.input_dimensions, self.second_input)
        self.bit_predictor = BitPredictor(self.bits)

    @nn.compact
    def __call__(self, stack: Array, actions: Array):
        bit_prediction = self.bit_predictor()
        pixels = self.autoencoder(stack, actions, bit_prediction)
        return pixels
