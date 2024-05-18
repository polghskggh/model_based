from src.models.initalizer.modelfreeinitializers import ActorInitializer, CriticInitializer, DQNInitializer
from src.models.initalizer.autoencoderinitializer import AutoEncoderInitializer
from src.models.initalizer.modelstrategy import ModelStrategy
from src.models.initalizer.trainerstrategy import InferenceInitializer, BitPredictorInitializer, StochasticInitializer, \
    StochasticAETrainerInitializer, LatentAEInitializer


def model_initializer_factory(strategy_type: str) -> ModelStrategy:
    match strategy_type:
        case "agent":
            return ActorInitializer()
        case "dqncritic":
            return DQNInitializer()
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
