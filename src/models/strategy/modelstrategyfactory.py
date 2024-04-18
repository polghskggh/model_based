from src.models.strategy.autoencoderstrategy import AutoEncoderStrategy
from src.models.strategy.ddpgactorstrategy import DDPGActorStrategy
from src.models.strategy.ddpgcriticstrategy import DDPGCriticStrategy
from src.models.strategy.modelstrategy import ModelStrategy
from src.models.strategy.trainerstrategy import InferenceStrategy, BitPredictorStrategy, StochasticAEStratgy, \
    StochasticAETrainerStrategy, LatentAutoEncoderStrategy


def model_strategy_factory(strategy_type: str) -> ModelStrategy:
    match strategy_type:
        case "actor":
            return DDPGActorStrategy()
        case "critic":
            return DDPGCriticStrategy()
        case "autoencoder":
            return AutoEncoderStrategy()
        case "stochastic_autoencoder":
            return StochasticAEStratgy()
        case "inference":
            return InferenceStrategy()
        case "bit_predictor":
            return BitPredictorStrategy()
        case "trainer_stochastic":
            return StochasticAETrainerStrategy()
        case "autoencoder_with_latent":
            return LatentAutoEncoderStrategy()
        case _:
            raise ValueError(f"Strategy {strategy_type} not found")
