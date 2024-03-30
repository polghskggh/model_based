import flax.linen as nn
import numpy as np


class SpatialAttention(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        # Define convolutions for Q, K, V projections

    def __call__(self, features):
        # Project features to different dimensions for Q, K, V
        query = nn.Conv(features, self.d_model, kernel_size=(1, 1))
        key = nn.Conv(features, self.d_model, kernel_size=(1, 1))
        value = nn.Conv(features, self.d_model, kernel_size=(1, 1))
        # Apply attention using softmax and weighted sum (similar to nn.attention)
        attention_weights = nn.softmax(query @ key.transpose((0, 2, 3, 1)) / np.sqrt(self.d_model))
        outputs = value @ attention_weights
        # Project back to original dimensions
        return nn.Conv(outputs, features.shape[-1], kernel_size=(1, 1))
