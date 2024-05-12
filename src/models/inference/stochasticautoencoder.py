import jax
import flax.linen as nn

from src.models.autoencoder.autoencoder import AutoEncoder
from src.models.inference.bitpredictior import BitPredictor


class StochasticAutoencoder(nn.Module):
    input_dimensions: tuple
    second_input: int
    bits: int = 128

    def setup(self):
        self.autoencoder = AutoEncoder(self.input_dimensions, self.second_input, deterministic=True)
        self.bit_predictor = BitPredictor(self.bits)

    @nn.compact
    def __call__(self, stack: jax.Array, actions: jax.Array):
        bit_predictions = self.bit_predictor(actions.shape[0])
        return self.autoencoder(stack, actions, bit_predictions)
