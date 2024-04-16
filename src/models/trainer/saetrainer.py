from ctypes import Array

from jax import random
from jax import random as jr

from src.agent.acstrategy import Shape
from src.models.atari.autoencoder.autoencoder import AutoEncoder
from src.models.atari.inference.bitpredictior import BitPredictor
from src.models.atari.inference.convolutionalinference import ConvolutionalInference
from src.models.atari.simple.trainsae import TrainStochasticAutoencoder
from src.models.modelwrapper import ModelWrapper
from src.models.trainer.trainer import Trainer


class SAETrainer(Trainer):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.autoencoder = ModelWrapper(AutoEncoder(*Shape()), "autoencoder")
        self.third_input = (Shape()[0][0], Shape()[0][1], 3)
        self.stochastic_ae = ModelWrapper(TrainStochasticAutoencoder(*Shape(), self.third_input), "trainerstochastic")
        self.bit_predictor = ModelWrapper(BitPredictor(model.bits).lstm, "bitpredictor")
        self.inference = ModelWrapper(ConvolutionalInference(*Shape(), self.third_input, False), "inference")
        self.batch_size = 32

    def train_step(self, params: dict, stack: Array, actions: Array, next_frame: Array):
        params = self.train_autoencoder(params, stack, actions, next_frame)
        params = self.train_inference_autoencoder(params, stack, actions, next_frame)
        params = self.train_inference_autoencoder(params, stack, actions, next_frame)
        params = self.train_predictor(params, stack, actions, next_frame)
        return params

    def train_autoencoder(self, params: dict, stack: Array, actions: Array, next_frame: Array):
        latent = jr.normal(random.PRNGKey(1), (self.batch_size, 128))
        self.autoencoder.params["params"]["autoencoder"] = params["params"]["autoencoder"]
        self.autoencoder.train_step(next_frame, stack, actions, latent)
        params["params"]["autoencoder"] = self.autoencoder.params["params"]["autoencoder"]
        return params

    def train_inference_autoencoder(self, params: dict, stack: Array, actions: Array, next_frame: Array):
        self.stochastic_ae.params["params"]["autoencoder"] = params["params"]["autoencoder"]
        self.stochastic_ae.train_step(next_frame, stack, actions, next_frame)
        params["params"]["autoencoder"] = self.stochastic_ae.params["params"]["autoencoder"]
        return params

    def train_predictor(self, params: dict, stack: Array, actions: Array, next_frame: Array):
        self.inference.params["params"] = params["params"]["inference"]

        bits_inferred = self.inference.forward(stack, actions, next_frame)
        self.bit_predictor.params["params"] = params["params"]["lstm"]
        last_bit = 0
        for inferred_bit in bits_inferred:
            self.bit_predictor.train_step(inferred_bit, last_bit)
            last_bit = inferred_bit

        params["params"]["lstm"] = self.bit_predictor.params["params"]
        return params
