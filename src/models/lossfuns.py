import jax.numpy as jnp
import optax
from optax import softmax_cross_entropy_with_integer_labels

from src.singletons.hyperparameters import Args


def mean_squared_error(state, params, teacher_outputs, *inputs, **kwargs):
    outputs = state.apply_fn(params, *inputs, **kwargs)
    return jnp.mean(optax.squared_error(outputs, teacher_outputs))  # output is not batch format


def image_loss_fn(pixels, teacher_pixels):
    teacher_pixels = jnp.squeeze(jnp.astype(teacher_pixels, jnp.int32))
    return jnp.mean(jnp.maximum(softmax_cross_entropy_with_integer_labels(pixels, teacher_pixels),
                                Args().args.pixel_loss_const))


def softmax_reward(reward, teacher_reward):
    teacher_reward = jnp.squeeze(jnp.astype(teacher_reward, jnp.int32))
    return softmax_cross_entropy_with_integer_labels(reward, teacher_reward)


def mse_reward(reward, teacher_reward):
    return optax.squared_error(reward.squeeze(), teacher_reward.squeeze())


def reward_loss_fn(reward, teacher_reward):
    if Args().args.rewards == 1:
        loss = mse_reward(reward, teacher_reward)
    else:
        loss = softmax_reward(reward, teacher_reward)
    return jnp.mean(loss)


def cross_entropy_loss(state, params, teacher_outputs, *inputs, **kwargs):
    teach_pixels, teach_reward = teacher_outputs
    teach_reward = jnp.astype(teach_reward, jnp.int32)
    pixels, reward = state.apply_fn(params, *inputs, **kwargs)
    alpha = Args().args.pixel_reward
    return alpha * image_loss_fn(pixels, teach_pixels) + (1 - alpha) * reward_loss_fn(reward, teach_reward)


def cross_entropy_with_kl_loss(state, params, teacher_outputs, *inputs, **kwargs):
    teach_pixels, teach_reward = teacher_outputs
    pixels, reward, kl_loss = state.apply_fn(params, *inputs, **kwargs)
    alpha = Args().args.pixel_reward
    beta = Args().args.kl_loss

    return alpha * (1 - beta) * image_loss_fn(pixels, teach_pixels) + alpha * beta * kl_loss \
        + (1 - alpha) * reward_loss_fn(reward, teach_reward)
