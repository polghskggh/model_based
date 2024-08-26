import flax.linen as nn
import jax.numpy as jnp
import jax.random as jr


class BitPredictor(nn.Module):
    features: int
    lstm: nn.Module = nn.OptimizedLSTMCell(features=1)

    def setup(self):
        rng = self.make_rng('carry')

    @nn.compact
    def __call__(self, batch_size: int) -> jnp.ndarray:
        prediction = jnp.zeros((batch_size, self.features))
        last_bit = jnp.zeros((batch_size, 1))
        carry = self.lstm.initialize_carry(self.make_rng('carry'), batch_size)

        for index in range(self.features):
            carry, last_bit = self.lstm(carry, last_bit)
            prediction = prediction.at[index].set(last_bit.item())
        return prediction
