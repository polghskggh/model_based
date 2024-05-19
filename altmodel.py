import string

import flax.linen as nn
from jax import Array
import jax.numpy as jnp
import gymnasium as gym
import numpy as np

from src.enviroment import Shape


class Network(nn.Module):
    input_dimensions: tuple
    def setup(self):
        self.actor = Actor(4)
        self.critic = Critic()

    @nn.compact
    def __call__(self, x: Array):
        x = nn.Sequential([
            convolution_layer_init(32, 8, 4),
            nn.relu,
            convolution_layer_init(64, 4, 2),
            nn.relu,
            convolution_layer_init(64, 3, 1),
            nn.relu,
        ])(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.Sequential([
            linear_layer_init(512),
            nn.relu
        ])(x)
        return self.actor(x), self.critic(x)


class Actor(nn.Module):
    action_n: int

    @nn.compact
    def __call__(self, x: Array):
        return linear_layer_init(self.action_n, std=0.01)(x)


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x: Array):
        return linear_layer_init(1, std=1)(x)

def make_env(env_id: string, idx: int, capture_video: bool, run_name: string):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode='rgb_array')
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{env_id}")
        else:
            env = gym.make(env_id, frameskip=1)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.AtariPreprocessing(env, grayscale_newaxis=True, scale_obs=True)
        Shape.initialize(env)
        return env

    return thunk


# Helper function to quickly declare linear layer with weight and bias initializers
def linear_layer_init(features, std=np.sqrt(2), bias_const=0.0):
    layer = nn.Dense(features=features, kernel_init=nn.initializers.orthogonal(std),
                     bias_init=nn.initializers.constant(bias_const))
    return layer


# Helper function to quickly declare convolution layer with weight and bias initializers
def convolution_layer_init(features, kernel_size, strides, std=np.sqrt(2), bias_const=0.0):
    layer = nn.Conv(features=features, kernel_size=(kernel_size, kernel_size), strides=(strides, strides), padding='VALID', kernel_init=nn.initializers.orthogonal(std), bias_init=nn.initializers.constant(bias_const))
    return layer

