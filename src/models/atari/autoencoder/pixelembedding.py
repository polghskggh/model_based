from flax import linen as nn
from jax import Array, vmap


# encoder
class PixelEmbedding(nn.Module):
    features: int
    mapped_dims: tuple

    def setup(self):
        self.embedding_layer = nn.Dense(features=self.features, name="embedding")
        self.mapped_layer = vmap(vmap(vmap(self.embedding_layer, self.mapped_dims), self.mapped_dims))

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = self.mapped_layer(x)
        return x



