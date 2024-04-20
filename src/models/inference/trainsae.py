from ctypes import Array

import flax.linen as nn

from src.models.autoencoder.autoencoder import AutoEncoder
from src.models.inference.convolutionalinference import ConvolutionalInference
from src.pod.hyperparameters import hyperparameters


class TrainStochasticAutoencoder(nn.Module):
    input_dimensions: tuple
    second_input: int
    third_input: tuple
    kl_divergence_weight: float = hyperparameters["world"]["kl_divergence_weight"]

    def setup(self):
        self.autoencoder = AutoEncoder(self.input_dimensions, self.second_input)
        self.inference = ConvolutionalInference(self.input_dimensions, self.second_input, self.third_input, True)

    @nn.compact
    def __call__(self, stack: Array, actions: Array, next_frame: Array, kl_divergence: bool = False):
        kl_loss = 0

        if kl_divergence:
            bit_prediction, kl_loss = self.inference(stack, actions, next_frame, True)
        else:
            bit_prediction = self.inference(stack, actions, next_frame)

        pixels = self.autoencoder(stack, actions, bit_prediction)

        return pixels, kl_loss * self.kl_divergence_weight

