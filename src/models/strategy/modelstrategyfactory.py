from src.models.strategy.actorstrategy import PPOActorStrategy
from src.models.strategy.autoencoderstrategy import AutoEncoderStrategy
from src.models.strategy.criticstrategy import DQNCriticStrategy, PPOCriticStrategy
from src.models.strategy.modelstrategy import ModelStrategy
from src.models.strategy.trainerstrategy import InferenceStrategy, BitPredictorStrategy, StochasticAEStratgy, \
    StochasticAETrainerStrategy, LatentAutoEncoderStrategy


def model_strategy_factory(strategy_type: str) -> ModelStrategy:
    match strategy_type:
        case "actor":
            return PPOActorStrategy()
        case "dqncritic":
            return DQNCriticStrategy()
        case "ppocritic":
            return PPOCriticStrategy()
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
