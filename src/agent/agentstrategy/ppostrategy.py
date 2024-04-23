import jax
import jax.numpy as jnp
from jax import tree_map

from src.agent.actor.ppoactor import PPOActor
from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.agent.critic.ppocritic import PPOCritic
from src.enviroment.shape import Shape
from src.models.actorcritic.actoratari import ActorAtari
from src.models.actorcritic.atarinn import AtariNN
from src.pod import ReplayBuffer
from src.pod.hyperparameters import hyperparameters
from src.utils.parammanip import sum_dicts


class PPOStrategy(StrategyInterface):
    def __init__(self):
        self._batch_size: int = hyperparameters["ppo"]['batch_size']
        self._batches_per_update: int = hyperparameters["ppo"]['batches_per_update']
        self._sequence_length: int = hyperparameters["ppo"]['sequence_length']
        self._actor, self._critic = PPOActor(ActorAtari(Shape()[0])), PPOCritic(AtariNN(*Shape(), 1))

    def _batch_update(self, training_sample: list[jax.Array]):
        states, rewards = jnp.append(training_sample[0], training_sample[3][-1]), training_sample[2]

        grads = self._critic.calculate_grads(states, rewards)
        self._critic.update(grads)
        advantage = self._critic.provide_feedback(states, rewards)
        grads = self._actor.calculate_grads(states, advantage)
        self._actor.update(grads)

    def update(self, replay_buffer: ReplayBuffer):
        # maybe do multiple updates per batch for PPO
        for _ in range(self._batches_per_update):
            self._batch_update(replay_buffer.sample(self._batch_size, self._sequence_length))

    def select_action(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._actor.calculate_action(state)

    def save(self):
        self._actor.save()
        self._critic.save()

    def load(self):
        self._actor.load()
        self._critic.load()
