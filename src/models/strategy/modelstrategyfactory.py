from src.models.strategy.autoencoderstrategy import AutoEncoderStrategy
from src.models.strategy.ddpgactorstrategy import DDPGActorStrategy
from src.models.strategy.ddpgcriticstrategy import DDPGCriticStrategy
from src.models.strategy.modelstrategy import ModelStrategy
from src.models.strategy.trainerstrategy import InferenceStrategy, BitPredictorStrategy, StochasticAEStratgy, \
    StochasticAETrainerStrategy


def model_strategy_factory(strategy_type: str) -> ModelStrategy:
    match strategy_type:
        case "actor":
            return DDPGActorStrategy()
        case "critic":
            return DDPGCriticStrategy()
        case "autoencoder":
            return AutoEncoderStrategy()
        case "stochasticautoencoder":
            return StochasticAEStratgy()
        case "inference":
            return InferenceStrategy()
        case "bitpredictor":
            return BitPredictorStrategy()
        case "trainerstochastic":
            return StochasticAETrainerStrategy()
        case _:
            raise ValueError(f"Strategy {strategy_type} not found")
