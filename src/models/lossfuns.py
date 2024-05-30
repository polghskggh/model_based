import jax
import jax.numpy as jnp
import optax
from optax import softmax_cross_entropy_with_integer_labels

from src.singletons.hyperparameters import Args


def mean_squared_error(state, params, teacher_outputs, *inputs, **kwargs):
    outputs = state.apply_fn(params, *inputs, **kwargs)
    return jnp.mean(optax.squared_error(outputs, teacher_outputs))  # output is not batch format


def cross_entropy_loss(state, params, teacher_outputs, *inputs, **kwargs):
    teach_pixels, teach_reward = teacher_outputs
    pixels, reward = state.apply_fn(params, *inputs, **kwargs)
    alpha = Args().args.pixel_reward
    return alpha * jnp.mean(softmax_cross_entropy_with_integer_labels(pixels, teach_pixels)) \
        + (1 - alpha) * jnp.mean(optax.softmax_cross_entropy_with_integer_labels(reward, teach_reward))


def cross_entropy_with_kl_loss(state, params, teacher_outputs, *inputs, **kwargs):
    teach_pixels, teach_reward = teacher_outputs
    pixel, reward, kl_loss = state.apply_fn(params, *inputs, **kwargs)
    alpha = Args().args.pixel_reward
    beta = Args().args.kl_loss

    return alpha * (1 - beta) * jnp.mean(softmax_cross_entropy_with_integer_labels(pixel, teach_pixels)) \
        + alpha * beta * kl_loss \
        + (1 - alpha) * jnp.mean(optax.squared_error(reward, teach_reward))
