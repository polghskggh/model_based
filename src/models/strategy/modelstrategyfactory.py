from src.models.strategy.autoencoderstrategy import AutoEncoderStrategy
from src.models.strategy.ddpgactorstrategy import DDPGActorStrategy
from src.models.strategy.ddpgcriticstrategy import DDPGCriticStrategy
from src.models.strategy.modelstrategy import ModelStrategy


def model_strategy_factory(strategy_type: str) -> ModelStrategy:
    if strategy_type == "actor":
        return DDPGActorStrategy()
    if strategy_type == "critic":
        return DDPGCriticStrategy()
    if strategy_type == "autoencoder":
        return AutoEncoderStrategy()
