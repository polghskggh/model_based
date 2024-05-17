from src.agent.agentstrategy.dqnstrategy import DQNStrategy
from src.agent.agentstrategy.ppostrategy import PPOStrategy
from src.agent.agentstrategy.strategyinterface import StrategyInterface


def agent_strategy_factory(strategy_name: str) -> StrategyInterface:
    """
    Factory method for creating agent strategies

    :param strategy_name: name of the initalizer
    :return: a new instance of the initalizer
    """
    match strategy_name:
        case "dqn":
            return DQNStrategy()
        case "ppo" | "dreamer" | "simple":
            return PPOStrategy()
        case _:
            raise ValueError(f"Unknown initializer: {strategy_name}")
