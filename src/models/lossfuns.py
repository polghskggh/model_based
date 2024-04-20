import jax.numpy as jnp
import optax
from optax import softmax_cross_entropy_with_integer_labels


def mean_squared_error(model, params, teacher_outputs, *inputs, **kwargs):
    outputs = model.apply(params, *inputs, **kwargs)
    return jnp.mean(optax.squared_error(outputs, teacher_outputs[0])) # output is not batch format


def cross_entropy_loss(model, params, teacher_outputs, *inputs, **kwargs):
    outputs = model.apply(params, *inputs, **kwargs)
    return jnp.mean(softmax_cross_entropy_with_integer_labels(outputs, teacher_outputs))


def cross_entropy_with_kl_loss(model, params, teacher_outputs, *inputs, **kwargs):
    outputs, kl_loss = model.apply(params, *inputs, **kwargs)
    return jnp.mean(softmax_cross_entropy_with_integer_labels(outputs, teacher_outputs)) + kl_loss
