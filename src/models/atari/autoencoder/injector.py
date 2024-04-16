import flax.linen as nn
from jax import Array


class Injector(nn.Module):
    source_features: int

    @nn.compact
    def __call__(self, x: Array, injection: Array) -> Array:
        inject_mul = nn.Dense(self.source_features, name="actions_mul")(injection)
        x *= nn.sigmoid(inject_mul)
        inject_add = nn.Dense(self.source_features, name="actions_add")(injection)
        x += inject_add
        return x