import jax
import jax.numpy as jnp
import optax
from optax import softmax_cross_entropy_with_integer_labels

from src.singletons.hyperparameters import Args


def mean_squared_error(state, params, teacher_outputs, *inputs, **kwargs):
    outputs = state.apply_fn(params, *inputs, **kwargs)
    return jnp.mean(optax.squared_error(outputs, teacher_outputs)), {}  # output is not batch format


def softmax_image_loss(pixels, teacher_pixels):
    teacher_pixels = jnp.squeeze(jnp.astype(teacher_pixels, jnp.int32))
    return jnp.maximum(softmax_cross_entropy_with_integer_labels(pixels, teacher_pixels),
                       Args().args.pixel_loss_const)


def mse_image_loss(pixels, teacher_pixels):
    return optax.squared_error(pixels, teacher_pixels)


def image_loss_fn(pixels, teacher_pixels):
    if Args().args.categorical_image:
        loss = softmax_image_loss(pixels, teacher_pixels)
    else:
        loss = mse_image_loss(pixels, teacher_pixels)
    return jnp.mean(loss)


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


def cross_entropy_with_dones(output, target):
    teach_pixels, teach_reward, teach_dones = target
    pixels, rewards, dones = output
    image_reward_loss, aux = cross_entropy_with_reward((pixels, rewards), (teach_pixels, teach_reward))
    dones_loss = jnp.mean(softmax_reward(dones, teach_dones))
    aux['dones_loss'] = dones_loss
    return image_reward_loss + dones_loss, aux


def cross_entropy_with_reward(output, target):
    teach_pixels, teach_reward = target
    pixels, rewards = output
    alpha = Args().args.pixel_reward
    image_loss = image_loss_fn(pixels, teach_pixels)
    reward_loss = reward_loss_fn(rewards, teach_reward)
    return alpha * image_loss + (1 - alpha) * reward_loss, {"image_loss": image_loss, "reward_loss": reward_loss}


def cross_entropy_loss(state, params, teacher_outputs, *inputs, **kwargs):
    output = state.apply_fn(params, *inputs, **kwargs)
    if Args().args.predict_dones:
        loss, aux = cross_entropy_with_dones(output, teacher_outputs)
    else:
        loss, aux = cross_entropy_with_reward(output, teacher_outputs)
    return loss, aux


def cross_entropy_with_kl_loss(state, params, teacher_outputs, *inputs, **kwargs):
    teach_pixels, teach_reward = teacher_outputs
    pixels, reward, kl_loss = state.apply_fn(params, *inputs, **kwargs)
    alpha = Args().args.pixel_reward
    beta = Args().args.kl_loss

    return alpha * (1 - beta) * image_loss_fn(pixels, teach_pixels) + alpha * beta * kl_loss \
        + (1 - alpha) * reward_loss_fn(reward, teach_reward)
