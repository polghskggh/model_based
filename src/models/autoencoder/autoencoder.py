import flax.linen as nn
import jax
from jax import Array, vmap
from rlax import one_hot
import jax.numpy as jnp

from src.models.autoencoder.decoder import Decoder
from src.models.autoencoder.encoder import Encoder
from src.models.autoencoder.injector import Injector
from src.models.autoencoder.logitslayer import LogitsLayer
from src.models.autoencoder.middlenetwork import MiddleNetwork
from src.models.autoencoder.rewardpredictor import Predictor
from src.models.helpers import linear_layer_init
from src.singletons.hyperparameters import Args


class AutoEncoder(nn.Module):
    input_dimensions: tuple
    second_input: int
    latent: int = 128
    deterministic: bool = True

    def setup(self):
        self.features = 256
        self.kernel = 4
        self.strides = 2
        self.layers = 6
        self.pixel_embedding = nn.Dense(features=self.features // 4, name="embedding")
        self.encoder = Encoder(self.features, self.kernel, self.strides, self.layers, self.deterministic)
        self.decoder = Decoder(self.features, self.kernel, self.strides, self.layers, self.deterministic)
        self.action_injector = Injector(self.features)
        self.latent_injector = Injector(self.features)
        self.middle_network = MiddleNetwork(self.features, self.kernel, self.deterministic)
        self.logits = LogitsLayer()
        self.reward_predictor = Predictor(Args().args.rewards)
        self.done_predictor = Predictor(2)

    @nn.compact
    def __call__(self, image: Array, action: Array, latent: Array = None):
        embedded_image = self.pixel_embedding(image)
        encoded, skip = self.encoder(embedded_image)
        encoded_action = one_hot(action, self.second_input)
        injected = self.action_injector(encoded, encoded_action)

        if latent is not None:
            injected = self.latent_injector(injected, latent)
        hidden = self.middle_network(injected)

        decoded = self.decoder(hidden, skip)
        logits = self.logits(decoded)

        reward_logits = self.reward_predictor(hidden, logits)
        if Args().args.rewards == 1:
            reward_logits = jnp.clip(reward_logits, -15, 15)

        if not Args().args.categorical_image:
            pixels = linear_layer_init(1 if Args().args.grayscale else 3)(logits)
            pixels = nn.relu(pixels)
        else:
            pixels = logits

        if Args().args.predict_dones:
            return pixels, reward_logits, self.done_predictor(hidden, logits)

        return pixels, reward_logits
