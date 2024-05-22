from ctypes import Array

import jax
from jax import random as jr
import jax.numpy as jnp

from src.enviroment import Shape
from src.models.autoencoder.autoencoder import AutoEncoder
from src.models.inference.bitpredictior import BitPredictor
from src.models.inference.convolutionalinference import ConvolutionalInference
from src.models.inference.trainsae import TrainStochasticAutoencoder
from src.models.modelwrapper import ModelWrapper
from src.singletons.hyperparameters import Args
from src.trainer.trainer import Trainer
from src.utils.rl import tile_image


class SAETrainer(Trainer):
    def __init__(self, model):
        self._batch_size = Args().args.batch_size

        self._model = model
        self._autoencoder = ModelWrapper(AutoEncoder(*Shape()), "autoencoder_with_latent")

        third_input = (Shape()[0][0], Shape()[0][1], 3)
        self._stochastic_ae = ModelWrapper(TrainStochasticAutoencoder(*Shape(), third_input), "trainer_stochastic")
        self._bit_predictor = ModelWrapper(BitPredictor(self._model.bits), "bit_predictor")

        self._inference = ModelWrapper(ConvolutionalInference(*Shape(), third_input, False), "inference")

        self._ae_trainer = ParamCopyingTrainer(self._autoencoder, "autoencoder")
        self._sae_trainer = ParamCopyingTrainer(self._stochastic_ae, "autoencoder", "autoencoder")
        self._bit_predictor_trainer = ParamCopyingTrainer(self._bit_predictor, "bit_predictor")

    def train_step(self, params: dict, stack: jax.Array, actions: jax.Array, rewards: jax.Array, next_frame: jax.Array):
        reconstructed = tile_image(next_frame)
        params = self._train_autoencoder(params, stack, actions, reconstructed)
        params = self._train_inference_autoencoder(params, stack, actions, next_frame, reconstructed)
        params = self._train_inference_autoencoder_with_kl(params, stack, actions, next_frame, reconstructed)
        params = self._train_predictor(params, stack, actions, next_frame)
        print("----------------------")
        return params

    def _train_autoencoder(self, params: dict, stack: Array, actions: Array, reconstructed: Array):
        latent = jnp.where(jr.normal(jr.PRNGKey(1), (stack.shape[0], 128)) >= 0, 1, 0)
        return self._ae_trainer.train_step(params, reconstructed, stack, actions, latent)

    def _train_inference_autoencoder(self, params: dict, stack: Array, actions: Array,
                                    next_frame: Array, reconstructed: Array):
        return self._sae_trainer.train_step(params, reconstructed, stack, actions, next_frame)

    def _train_inference_autoencoder_with_kl(self, params: dict, stack: Array, actions: Array,
                                     next_frame: Array, reconstructed: Array):
        self._sae_trainer.train_step(self._stochastic_ae.params, reconstructed, stack,
                                     actions, next_frame, True)
        return params

    def _train_predictor(self, params: dict, stack: Array, actions: Array, next_frame: Array):
        self._inference.params["params"] = self._stochastic_ae.params["params"]["inference"]
        bits_inferred = self._inference.forward(stack, actions, next_frame)
        return self._bit_predictor_trainer.train_step(params, bits_inferred)

class ParamCopyingTrainer(Trainer):
    def __init__(self, model, param_name, model_param_name=None):
        super().__init__()
        self._model = model
        self._param_name = param_name
        self._model_param_name = model_param_name

    def train_step(self, params: dict, output, *inputs):
        ParamCopyingTrainer.assign_params(self._model.params, self._model_param_name, params, self._param_name)

        batch_size = Args().args.batch_size

        for start_idx in range(0, output.shape[0], batch_size):
            batch_slice = slice(start_idx, start_idx + batch_size)
            grads = self._model.train_step(output[batch_slice], *inputs[batch_slice])
            self._model.apply_grads(grads)

        ParamCopyingTrainer.assign_params(params, self._param_name, self._model.params, self._model_param_name)
        return params

    @staticmethod
    def assign_params(params_to, name_to, params_from, name_from):
        if name_from is None:
            params_to["params"][name_to] = params_from["params"]
        elif name_to is None:
            params_to["params"] = params_from["params"][name_from]
        else:
            params_to["params"][name_to] = params_from["params"][name_from]


