from ctypes import Array

import jax
from jax import tree_map

from src.agent.actor.ppoactor import DDPGActor
from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.agent.critic import DDPGCritic
from src.agent.trajectory.trajectoryinterface import TrajectoryInterface
from src.enviroment.shape import Shape
from src.models.actorcritic.actoratari import ActorAtari
from src.models.actorcritic.atarinn import AtariNN
from src.pod import ReplayBuffer
from src.pod.hyperparameters import hyperparameters
from src.utils.parammanip import sum_dicts

import jax.numpy as jnp


class DDPGStrategy(StrategyInterface):
    def __init__(self):
        self._batch_size: int = hyperparameters["ddpg"]['batch_size']
        self._batches_per_update: int = hyperparameters["ddpg"]['batches_per_update']
        self._actor, self._critic = DDPGActor(ActorAtari(Shape()[0])), DDPGCritic(AtariNN(*Shape()))

    def _batch_update(self, training_sample: list[jax.Array]) -> tuple[jax.Array, jax.Array]:
        actor_actions = self._actor.calculate_actions(training_sample[2])

        critic_grads = self._critic.calculate_grads(
            training_sample[0], training_sample[1], training_sample[3],
            training_sample[2], actor_actions)

        action_grads = self._critic.provide_feedback(training_sample[2], actor_actions)
        return critic_grads, action_grads

    def update(self, replay_buffer: ReplayBuffer):
        critic_grads_sum, actor_grads_sum = self._batch_update(replay_buffer.sample(self._batch_size))

        # maybe do multiple updates per batch for PPO
        for _ in range(self._batches_per_update - 1):
            critic_grads, actor_grads = self._batch_update(replay_buffer.sample(self._batch_size))
            critic_grads_sum = sum_dicts(critic_grads_sum, critic_grads)
            actor_grads_sum = sum_dicts(actor_grads_sum, actor_grads)

        self._critic.update(tree_map(lambda x: x / self._batches_per_update, critic_grads_sum))
        self._actor.update(tree_map(lambda x: -x / self._batches_per_update, actor_grads_sum))

    def select_action(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._actor.approximate_best_action(state)
