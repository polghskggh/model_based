import flax.linen as nn
from jax import Array

from altmodel import linear_layer_init


class Injector(nn.Module):
    source_features: int

    @nn.compact
    def __call__(self, x: Array, injection: Array) -> Array:
        injection = injection.reshape((-1, 1, 1, injection.shape[-1]))
        inject_mul = linear_layer_init(self.source_features)(injection)
        x *= nn.sigmoid(inject_mul)
        inject_add = linear_layer_init(self.source_features)(injection)
        x += inject_add
        return x