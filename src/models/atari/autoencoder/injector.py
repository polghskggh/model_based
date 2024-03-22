import flax.linen as nn
from jax import Array


class Injector(nn.Module):
    @nn.compact
    def __call__(self, x: Array, injection: Array) -> Array:
        features = x.shape[-1]
        injection = injection.reshape([-1, 1, 1, injection.shape[-1]])
        inject_mul = nn.Dense(features, name="actions_mul")(injection)
        x *= nn.sigmoid(inject_mul)
        inject_add = nn.Dense(features, name="actions_add")(injection)
        x += inject_add
        return x
    # TODO: make it work on batches