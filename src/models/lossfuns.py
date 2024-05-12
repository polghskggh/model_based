import jax.debug
import jax.numpy as jnp
import optax
from jax import jit
from optax import softmax_cross_entropy_with_integer_labels

from src.pod.hyperparameters import hyperparameters


def mean_squared_error(model, params, teacher_outputs, *inputs, **kwargs):
    outputs = jit(model.apply)(params, *inputs, **kwargs)
    return jnp.mean(optax.squared_error(outputs, teacher_outputs))  # output is not batch format


def cross_entropy_loss(model, params, teacher_outputs, *inputs, **kwargs):
    teach_pixels, teach_reward = teacher_outputs
    pixels, reward = jit(model.apply)(params, *inputs, **kwargs)
    alpha = hyperparameters["simple"]["pixel_reward"]
    print(pixels.shape, teach_pixels.shape)
    return alpha * jnp.mean(softmax_cross_entropy_with_integer_labels(pixels, teach_pixels)) \
        + (1 - alpha) * jnp.mean(optax.squared_error(reward, teach_reward))


def cross_entropy_with_kl_loss(model, params, teacher_outputs, *inputs, **kwargs):
    teach_pixels, teach_reward = teacher_outputs
    pixel, reward, kl_loss = jit(model.apply)(params, *inputs, **kwargs)
    alpha = hyperparameters["simple"]["pixel_reward"]
    beta = hyperparameters["simple"]["kl_loss"]

    return alpha * (1 - beta) * jnp.mean(softmax_cross_entropy_with_integer_labels(pixel, teach_pixels)) \
        + alpha * beta * kl_loss \
        + (1 - alpha) * jnp.mean(optax.squared_error(reward, teach_reward))
