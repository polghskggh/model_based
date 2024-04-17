import jax.numpy as jnp


def mean_squared_error(model, params, teacher_outputs, *inputs):
    return jnp.mean((model.apply(params, *inputs) - teacher_outputs) ** 2)


def cross_entropy_loss(model, params, teacher_outputs, *inputs):
    return -jnp.mean(teacher_outputs * jnp.log(model.apply(params, *inputs)))