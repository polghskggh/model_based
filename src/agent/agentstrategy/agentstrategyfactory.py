from src.agent.agentstrategy.ddpgstrategy import DDPGStrategy
from src.agent.agentstrategy.dqnstrategy import DQNStrategy
from src.agent.agentstrategy.ppostrategy import PPOStrategy
from src.agent.agentstrategy.strategyinterface import StrategyInterface


def agent_strategy_factory(strategy_name: str) -> StrategyInterface:
    """
    Factory method for creating agent strategies

    :param strategy_name: name of the strategy
    :return: a new instance of the strategy
    """
    match strategy_name:
        case "ddpg":
            return DDPGStrategy()
        case "dqn":
            return DQNStrategy()
        case "ppo":
            return PPOStrategy()
        case _:
            raise ValueError(f"Unknown strategy: {strategy_name}")
