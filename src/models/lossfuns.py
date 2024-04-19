import jax.numpy as jnp
from optax.losses import kl_divergence


def mean_squared_error(model, params, teacher_outputs, *inputs, **kwargs):
    return jnp.mean((model.apply(params, *inputs, **kwargs) - teacher_outputs) ** 2)


def cross_entropy_loss(model, params, teacher_outputs, *inputs, **kwargs):
    outputs = model.apply(params, *inputs, **kwargs)
    return -jnp.mean(teacher_outputs * jnp.log(outputs))


def cross_entropy_with_kl_loss(model, params, teacher_outputs, *inputs, **kwargs):
    outputs, kl_loss = model.apply(params, *inputs, **kwargs)
    return -jnp.mean(teacher_outputs * jnp.log(outputs)) + kl_loss
