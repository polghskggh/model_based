import flax.linen as nn
from jax import Array, vmap

from src.models.atari.autoencoder.decoder import Decoder
from src.models.atari.autoencoder.encoder import Encoder
from src.models.atari.autoencoder.injector import Injector
from src.models.atari.autoencoder.logitslayer import LogitsLayer
from src.models.atari.autoencoder.pixelembedding import PixelEmbedding
from src.models.atari.autoencoder.rewardpredictor import RewardPredictor


class AutoEncoder(nn.Module):
    input_dimensions: tuple
    second_input: int

    def setup(self):
        self.features = 256
        self.kernel = (4, 4)
        self.strides = (2, 2)
        self.layers = 6
        self.pixel_embedding = PixelEmbedding(64)
        self.encoder = Encoder(self.features, self.kernel, self.strides, self.layers)
        self.decoder = Decoder(self.features, self.kernel, self.strides, self.layers)
        self.injector = Injector()
        self.logits = LogitsLayer()
        self.reward_predictor = RewardPredictor()
        self.softmax = vmap(vmap(nn.softmax))

    @nn.compact
    def __call__(self, image: Array, action: Array):
        image = self.turn_into_batch(image)
        action = self.turn_into_batch(action)

        embedded_image = self.pixel_embedding(image)
        encoded, skip = self.encoder(embedded_image)
        injected = self.injector(encoded, action)
        decoded = self.decoder(injected, skip)
        logits = self.logits(decoded)
        #reward = self.reward_predictor(injected, logits)
        pixels = self.softmax( logits)
        return pixels

    @staticmethod
    def turn_into_batch(x: Array) -> Array:
        return x.reshape(1, *x.shape) if x.ndim == 3 else x
