import jax.numpy as jnp
import optax
from optax import softmax_cross_entropy_with_integer_labels

from src.pod.hyperparameters import hyperparameters


def mean_squared_error(model, params, teacher_outputs, *inputs, **kwargs):
    outputs = model.apply(params, *inputs, **kwargs)
    print(outputs.shape, teacher_outputs.shape)
    return jnp.mean(optax.squared_error(outputs, teacher_outputs))  # output is not batch format


def cross_entropy_loss(model, params, teacher_outputs, *inputs, **kwargs):
    outputs = model.apply(params, *inputs, **kwargs)
    return jnp.mean(softmax_cross_entropy_with_integer_labels(outputs, teacher_outputs))


def grad_ascend(model, params, *inputs, **kwargs):
    return jnp.mean(model.apply(params, *inputs, **kwargs))


def cross_entropy_with_kl_loss(model, params, teacher_outputs, *inputs, **kwargs):
    outputs, kl_loss = model.apply(params, *inputs, **kwargs)
    alpha = hyperparameters["mixing_coefficients"]["kl_loss"]
    return (1 - alpha) * jnp.mean(softmax_cross_entropy_with_integer_labels(outputs, teacher_outputs)) + alpha * kl_loss
