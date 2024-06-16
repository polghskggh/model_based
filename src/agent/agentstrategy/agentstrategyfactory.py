from src.agent.agentstrategy.dqnstrategy import DQNStrategy
from src.agent.agentstrategy.ppostrategy import PPOStrategy
from src.agent.agentstrategy.strategyinterface import StrategyInterface
import jax.random as jr
import jax.numpy as jnp

from src.enviroment import Shape
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key


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
        case "random":
            return RandomStrategy()
        case _:
            raise ValueError(f"Unknown initializer: {strategy_name}")


class RandomStrategy(StrategyInterface):
    def select_action(self, *args, **kwargs):
        return jr.randint(Key().key(), shape=(Args().args.num_envs,), minval=0, maxval=Shape()[1])

    def timestep_callback(self, old_state: jnp.ndarray, selected_action: int, reward: float, new_state: jnp.ndarray,
                          done: bool, store_trajectory: bool):
        pass
