from ctypes import Array

import flax.linen as nn

from src.models.atari.autoencoder.autoencoder import AutoEncoder
from src.models.atari.inference.bitpredictior import BitPredictor


class StochasticAutoEncoder(nn.Module):
    def setup(self):
        self.autoencoder = AutoEncoder()
        self.bit_predictor = BitPredictor(128)

    @nn.compact
    def __call__(self, stack: Array, actions: Array):
        bit_prediction = self.bit_predictor()
        pixels = self.autoencoder(stack, actions, bit_prediction)
        return pixels