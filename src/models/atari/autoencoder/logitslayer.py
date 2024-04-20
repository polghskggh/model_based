from flax import linen as nn
from jax import Array, vmap


# logits
class LogitsLayer(nn.Module):
    def setup(self):
        self.rgb = 256
        self.final_layer = nn.Dense(features=self.rgb, name="logits")
        self.mapped_layer = vmap(vmap(self.final_layer))

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = self.mapped_layer(x)
        return x



