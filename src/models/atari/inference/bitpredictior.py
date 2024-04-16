import flax.linen as nn
import jax.numpy as jnp
import jax.random as jr


class BitPredictor(nn.Module):
    features: int
    lstm: nn.Module = nn.OptimizedLSTMCell(features=1)

    def setup(self):
        self.last_bit = jnp.array([0])
        self.carry = self.lstm.initialize_carry(jr.PRNGKey(0), self.last_bit.shape)

    @nn.compact
    def __call__(self):
        prediction = jnp.zeros(self.features)
        last_bit = self.last_bit
        carry = self.carry

        for index in range(self.features):
            carry, last_bit = self.lstm(carry, last_bit)
            prediction = prediction.at[index].set(last_bit.item())
        return prediction
