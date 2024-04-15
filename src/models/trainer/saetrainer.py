from ctypes import Array

import optax
from jax import value_and_grad, random

from src.models.atari.autoencoder.autoencoder import AutoEncoder
from src.models.atari.inference.bitpredictior import BitPredictor
from src.models.atari.inference.convolutionalinference import ConvolutionalInference
from src.models.atari.simple.trainsae import TrainStochasticAutoencoder
from src.models.lossfuns import mean_squared_error, cross_entropy_loss
from src.models.trainer.trainer import Trainer
import jax.numpy as jnp


class SAETrainer(Trainer):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.bit_predictor = BitPredictor(model.bits).lstm
        self.inference = ConvolutionalInference()
        self.stochastic_autoencoder = TrainStochasticAutoencoder()

        self._optimizer = optax.adam(0.0001)
        self.stochastic_autoencoder_params = self.stochastic_autoencoder.init_params(random.PRNGKey(1),
                                                  *(jnp.ones(model.input_dimensions, dtype=jnp.float32),
                                                    jnp.ones(model.second_input, dtype=jnp.float32),
                                                    jnp.ones(model.third_input, dtype=jnp.float32)))
        self.stochastic_autoencoder_state = self._optimizer.init(self.stochastic_autoencoder_params)

    def train_step(self, params: dict, stack: Array, actions: Array, next_frame: Array):
        params = self.train_autoencoder(params, stack, actions, next_frame)

        bits_inferred = self.inference(stack, actions, next_frame)
        params = self.train_predictor(params, bits_inferred)

        return params

    def train_autoencoder(self, params, stack, actions, next_frame):
        # params are not correct
        loss, grads = value_and_grad(cross_entropy_loss)(self.stochastic_autoencoder,
                                                         self.stochastic_autoencoder_params,
                                                         stack, actions, next_frame)
        print("autoencoder loss: ", loss)
        opt_grads, self.stochastic_autoencoder_state = self._optimizer.update(grads, self.stochastic_autoencoder_state)
        self.stochastic_autoencoder_params = optax.apply_updates(self.stochastic_autoencoder_params, opt_grads)
        params["params"]["autoencoder"] = self.stochastic_autoencoder_params["params"]["autoencoder"]
        return params

    def train_predictor(self, params: dict, bits_inferred: Array):
        print(params)
        lstm_params = params["params"]["lstm"]
        last_bit = 0
        for bit_inferred in bits_inferred:
            loss, grads = value_and_grad(mean_squared_error, 1)(self.bit_predictor, lstm_params, bit_inferred, last_bit)
            print("lstm loss: ", loss)
            last_bit = bit_inferred

        params["params"]["lstm"] = lstm_params
        return params