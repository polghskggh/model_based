from ctypes import Array

from jax import random as jr

from src.enviroment import Shape
from src.models.autoencoder.autoencoder import AutoEncoder
from src.models.inference.bitpredictior import BitPredictor
from src.models.inference.convolutionalinference import ConvolutionalInference
from src.models.inference.trainsae import TrainStochasticAutoencoder
from src.models.modelwrapper import ModelWrapper
from src.trainer.trainer import Trainer
from src.pod.hyperparameters import hyperparameters
from src.utils.tiling import tile_image


class SAETrainer(Trainer):
    def __init__(self, model):
        super().__init__()
        self._batch_size = 32

        self._model = model
        self._autoencoder = ModelWrapper(AutoEncoder(*Shape()), "autoencoder_with_latent",
                                         learning_rate=hyperparameters["world"]["deterministic_lr"])

        third_input = (Shape()[0][0], Shape()[0][1], 3)
        self._stochastic_ae = ModelWrapper(TrainStochasticAutoencoder(*Shape(), third_input), "trainer_stochastic",
                                           learning_rate=hyperparameters["world"]["stochastic_lr"])
        self._bit_predictor = ModelWrapper(BitPredictor(model.bits), "bit_predictor",
                                           learning_rate=hyperparameters["world"]["bit_predictor_lr"])

        self._inference = ModelWrapper(ConvolutionalInference(*Shape(), third_input, False), "inference",
                                       learning_rate=0)

        self._ae_trainer = ParamCopyingTrainer(self._autoencoder, "autoencoder")
        self._sae_trainer = ParamCopyingTrainer(self._stochastic_ae, "autoencoder", "autoencoder")
        self._bit_predictor_trainer = ParamCopyingTrainer(self._bit_predictor, "bit_predictor")

    def train_step(self, params: dict, stack: Array, actions: Array, next_frame: Array):
        reconstructed = tile_image(next_frame)
        params = self._train_autoencoder(params, stack, actions, reconstructed)
        params = self._train_inference_autoencoder(params, stack, actions, next_frame, reconstructed)
        params = self._train_inference_autoencoder_with_kl(params, stack, actions, next_frame, reconstructed)
        params = self._train_predictor(params, stack, actions, next_frame)
        print("----------------------")
        return params

    def _train_autoencoder(self, params: dict, stack: Array, actions: Array, reconstructed: Array):
        latent = jr.normal(jr.PRNGKey(1), (self._batch_size, 128))
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

        grads = self._model.train_step(output, *inputs)
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


