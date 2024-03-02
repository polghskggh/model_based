from src.agent.critic import DDPGCritic
from src.models import MLPSimple
from src.agent.actor import DDPGActor
from src.models.atari import AtariNN
from src.models.atari.actoratari import ActorAtari

strategy = {
    "ddpg": (DDPGActor(MLPSimple(24, 4), polyak=0.95),
             DDPGCritic(MLPSimple(28, 1), discount_factor=0.95, polyak=0.975, action_dim=4)),
    "atari-ddpg": (DDPGActor(ActorAtari((255, 100, 3), 4), polyak=0.95),
                   DDPGCritic(AtariNN((255, 100, 3), 4, 1), discount_factor=0.95, polyak=4, action_dim=4))
}


