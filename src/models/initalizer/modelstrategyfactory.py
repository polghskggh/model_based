from src.models.initalizer.actorstrategy import ActorInitializer
from src.models.initalizer.autoencoderinitializer import AutoEncoderInitializer
from src.models.initalizer.criticstrategy import DQNCriticStrategy, CriticInitializer
from src.models.initalizer.modelstrategy import ModelStrategy
from src.models.initalizer.trainerstrategy import InferenceInitializer, BitPredictorInitializer, StochasticInitializer, \
    StochasticAETrainerInitializer, LatentAEInitializer


def model_initializer_factory(strategy_type: str) -> ModelStrategy:
    match strategy_type:
        case "actor":
            return ActorInitializer()
        case "dqncritic":
            return DQNCriticStrategy()
        case "ppocritic":
            return CriticInitializer()
        case "autoencoder":
            return AutoEncoderInitializer()
        case "stochastic_autoencoder":
            return StochasticInitializer()
        case "inference":
            return InferenceInitializer()
        case "bit_predictor":
            return BitPredictorInitializer()
        case "trainer_stochastic":
            return StochasticAETrainerInitializer()
        case "autoencoder_with_latent":
            return LatentAEInitializer()
        case _:
            raise ValueError(f"Initalizetion {strategy_type} not found")
