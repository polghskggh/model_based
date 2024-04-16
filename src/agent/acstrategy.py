from typing import Self

from src.agent.actor.ddpgactor import DDPGActor
from src.agent.critic import DDPGCritic
from src.models.atari import AtariNN
from src.models.atari.actoratari import ActorAtari


class Shape:
    shape: tuple = None

    def __new__(cls, env=None) -> Self:
        if env is not None:
            cls.shape = (env.observation_space.shape, env.action_space.n)
        return cls.shape
