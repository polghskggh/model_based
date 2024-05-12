import flax.linen as nn
from jax import Array, vmap
from rlax import one_hot

from src.models.autoencoder.decoder import Decoder
from src.models.autoencoder.encoder import Encoder
from src.models.autoencoder.injector import Injector
from src.models.autoencoder.logitslayer import LogitsLayer
from src.models.autoencoder.rewardpredictor import RewardPredictor


class AutoEncoder(nn.Module):
    input_dimensions: tuple
    second_input: int
    latent: int = 128
    deterministic: bool = False

    def setup(self):
        self.features = 256
        self.kernel = (4, 4)
        self.strides = (2, 2)
        self.layers = 6
        self.pixel_embedding = nn.Dense(features=self.features // 4, name="embedding")
        self.encoder = Encoder(self.features, self.kernel, self.strides, self.layers, self.deterministic)
        self.decoder = Decoder(self.features, self.kernel, self.strides, self.layers, self.deterministic)
        self.action_injector = Injector(self.features)
        self.latent_injector = Injector(self.features)
        self.logits = LogitsLayer()
        self.reward_predictor = RewardPredictor()
        self.softmax = vmap(vmap(nn.softmax))

    @nn.compact
    def __call__(self, image: Array, action: Array, latent: Array = None) -> Array:
        embedded_image = self.pixel_embedding(image)
        encoded, skip = self.encoder(embedded_image)
        encoded_action = one_hot(action, self.second_input)
        injected = self.action_injector(encoded, encoded_action)

        if latent is not None:
            injected = self.latent_injector(encoded, latent)

        decoded = self.decoder(injected, skip)
        logits = self.logits(decoded)
        reward = self.reward_predictor(injected, logits)
        pixels = self.softmax(logits)
        return pixels, reward

