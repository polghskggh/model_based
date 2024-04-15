import flax.linen as nn
import jax.numpy as jnp


class BitPredictor(nn.Module):
    features: int

    def setup(self):
        self.lstm = nn.LSTMCell(features=self.features)
        self.last_bit = 0

    @nn.compact
    def __call__(self):
        prediction = jnp.zeros(self.features)
        for index in range(self.features):
            self.last_bit = self.lstm(self.last_bit)
            prediction[index] = self.last_bit

        return prediction