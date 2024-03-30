import flax.linen as nn
import jax.numpy as jnp
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import default_kernel_init
from flax.typing import Initializer, Dtype


class Attention(nn.Module):
    kernel_init: Initializer = default_kernel_init
    param_dtype: Dtype = jnp.float32

    def __call__(self, inputs):
        kernel = self.param("attention",
                            self.kernel_init,
                            (jnp.shape(inputs)[-1], self.features),
                            self.param_dtype)

        inputs, kernel = promote_dtype(inputs, kernel, dtype=self.d_type)
        return inputs * kernel
