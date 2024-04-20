from src.agent.trajectory.offpolicy import OffPolicy
from src.agent.trajectory.onpolicy import OnPolicy
from src.agent.trajectory.trajectoryinterface import TrajectoryInterface


def trajectory_factory(strategy_name: str) -> TrajectoryInterface:
    if strategy_name in ["ppo"]:
        return OnPolicy()
    elif strategy_name in ["ddpg", "dqn"]:
        return OffPolicy()

    raise ValueError(f"Unknown strategy: {strategy_name}")
