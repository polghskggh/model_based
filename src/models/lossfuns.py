import jax.numpy as jnp
from optax.losses import kl_divergence


def mean_squared_error(model, params, teacher_outputs, *inputs, **kwargs):
    return jnp.mean((model.apply(params, *inputs, **kwargs) - teacher_outputs) ** 2)


def cross_entropy_loss(model, params, teacher_outputs, *inputs, **kwargs):
    return -jnp.mean(teacher_outputs * jnp.log(model.apply(params, *inputs, **kwargs)))


def cross_entropy_loss_with_kl(model, params, teacher_outputs, *inputs, **kwargs):
    latent, output = model.apply(params, *inputs, **kwargs)
    output_loss = -jnp.mean(teacher_outputs * jnp.log(output))
    kl_loss = kl_divergence(latent, output) if latent is not None else 0
    return output_loss + kl_loss
