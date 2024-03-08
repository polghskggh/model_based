from src.agent.critic import DDPGCritic
from src.agent.actor import DDPGActor
from src.models.atari import AtariNN
from src.models.atari.actoratari import ActorAtari
from math import prod

shapes = {
    "atari-ddpg": ((210, 160, 3), 4)
}


strategy = {
    "atari-ddpg": (DDPGActor(ActorAtari(shapes["atari-ddpg"][0], shapes["atari-ddpg"][1]), polyak=0.95),

                   DDPGCritic(AtariNN((shapes["atari-ddpg"][0], shapes["atari-ddpg"][1]), 1),
                              discount_factor=0.95, polyak=4, action_dim=4))
}

