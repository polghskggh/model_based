from src.enviroment import Shape
from src.models.autoencoder.autoencoder import AutoEncoder
from src.models.inference.stochasticautoencoder import StochasticAutoencoder
from src.models.modelwrapper import ModelWrapper
from src.models.trainer.saetrainer import SAETrainer
from src.worldmodel.worldmodelinterface import WorldModelInterface


class SimpleWorldModel(WorldModelInterface):
    def __init__(self):
        self._model: ModelWrapper = ModelWrapper(StochasticAutoencoder(*Shape()), "stochastic_autoencoder")
        self.trainer = SAETrainer(self._model.model)

