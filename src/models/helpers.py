import flax.linen as nn
import numpy as np


# Helper function to quickly declare linear layer with weight and bias initializers
def linear_layer_init(features, std=np.sqrt(2), bias_const=0.0):
    layer = nn.Dense(features=features,
                     kernel_init=nn.initializers.orthogonal(std),
                     bias_init=nn.initializers.constant(bias_const))
    return layer


# Helper function to quickly declare convolution layer with weight and bias initializers
def convolution_layer_init(features, kernel_size, strides, std=np.sqrt(2), bias_const=0.0):
    layer = nn.Conv(features=features, kernel_size=(kernel_size, kernel_size), strides=(strides, strides),
                    padding='VALID', kernel_init=nn.initializers.orthogonal(std),
                    bias_init=nn.initializers.constant(bias_const))
    return layer
