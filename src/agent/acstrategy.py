from src.agent.critic import DDPGCritic
from src.models import MLPSimple
from src.agent.actor import DDPGActor

strategy = {
    "DDPG": (DDPGActor(MLPSimple(24, 4), polyak=0.95),
             DDPGCritic(MLPSimple(28, 1), discount_factor=0.95, polyak=0.975, action_dim=4)),

}