from ctypes import Array

import jax.numpy as jnp
from jax import random as jr

from src.agent.acstrategy import Shape
from src.models.atari.autoencoder.autoencoder import AutoEncoder
from src.models.atari.inference.bitpredictior import BitPredictor
from src.models.atari.inference.convolutionalinference import ConvolutionalInference
from src.models.atari.simple.trainsae import TrainStochasticAutoencoder
from src.models.modelwrapper import ModelWrapper
from src.models.trainer.trainer import Trainer
from src.utils.inttoonehot import tiles_to_onehot
from src.utils.tiling import tile_image


class SAETrainer(Trainer):
    def __init__(self, model):
        super().__init__()
        self._model = model
        self._autoencoder = ModelWrapper(AutoEncoder(*Shape()), "autoencoder")
        third_input = (Shape()[0][0], Shape()[0][1], 3)
        self._stochastic_ae = ModelWrapper(TrainStochasticAutoencoder(*Shape(), third_input), "trainerstochastic")
        self._bit_predictor = ModelWrapper(BitPredictor(model.bits), "bitpredictor")
        self._inference = ModelWrapper(ConvolutionalInference(*Shape(), third_input, False), "inference")
        self._batch_size = 32

    def train_step(self, params: dict, stack: Array, actions: Array, next_frame: Array):
        reconstructed = tiles_to_onehot(tile_image(next_frame))
        params = self._train_autoencoder(params, stack, actions, reconstructed)
        params = self._train_inference_autoencoder(params, stack, actions, next_frame, reconstructed)
        params = self._train_inference_autoencoder(params, stack, actions, next_frame, reconstructed)
        params = self._train_predictor(params, stack, actions, next_frame)
        return params

    def _train_autoencoder(self, params: dict, stack: Array, actions: Array, reconstructed: Array):
        latent = jr.normal(jr.PRNGKey(1), (self._batch_size, 128))
        self._autoencoder.params["params"]["autoencoder"] = params["params"]["autoencoder"]
        self._autoencoder.train_step(reconstructed, stack, actions, latent)
        params["params"]["autoencoder"] = self._autoencoder.params["params"]["autoencoder"]
        return params

    def _train_inference_autoencoder(self, params: dict, stack: Array, actions: Array,
                                    next_frame: Array, reconstructed: Array):
        self._stochastic_ae.params["params"]["autoencoder"] = params["params"]["autoencoder"]
        self._stochastic_ae.train_step(reconstructed, stack, actions, next_frame)
        params["params"]["autoencoder"] = self._stochastic_ae.params["params"]["autoencoder"]
        return params

    def _train_predictor(self, params: dict, stack: Array, actions: Array, next_frame: Array):
        self._inference.params["params"] = self._stochastic_ae.params["params"]["inference"]

        bits_inferred = self._inference.forward(stack, actions, next_frame)
        self._bit_predictor.params["params"] = params["params"]["bit_predictor"]

        self._bit_predictor.train_step(bits_inferred)

        params["params"]["bit_predictor"]["lstm"] = self._bit_predictor.params["params"]
        return params
