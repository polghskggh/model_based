import jax
import jax.numpy as jnp
from jax import tree_map, lax

from src.agent.actor.ppoactor import PPOActor
from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.agent.critic.ppocritic import PPOCritic
from src.enviroment.shape import Shape
from src.models.actorcritic.actoratari import ActorAtari
from src.models.actorcritic.atarinn import StateValueAtariNN
from src.pod.hyperparameters import hyperparameters
from src.pod.replaybuffer import ReplayBuffer
from src.utils.inttoonehot import softmax_to_onehot
from src.utils.parammanip import sum_dicts


class PPOStrategy(StrategyInterface):
    def __init__(self):
        self._batch_size: int = hyperparameters['ppo']['batch_size']
        self._sequence_length: int = hyperparameters['ppo']['sequence_length']
        self._actor, self._critic = PPOActor(ActorAtari(*Shape())), PPOCritic(StateValueAtariNN(Shape()[0], 1))

    def _batch_update(self, training_sample: list[jax.Array]):
        final_state = training_sample[3]
        final_state = lax.slice_in_dim(final_state, self._sequence_length - 1, None, axis=1)

        extended_states = jnp.append(training_sample[0], final_state, axis=1)

        action = training_sample[1]
        rewards = training_sample[2]

        advantage = self._critic.provide_feedback(extended_states, rewards)

        states = training_sample[0]
        actor_grads = self._actor.calculate_grads(states, advantage, action)
        critic_grads = self._critic.calculate_grads(extended_states, rewards)
        return actor_grads, critic_grads

    def update(self, replay_buffer: ReplayBuffer):
        transitions = replay_buffer.sample(self._batch_size, self._sequence_length)
        actor_grads, critic_grads = self._batch_update(transitions)

        self._actor.update(actor_grads)
        self._critic.update(critic_grads)

    def action_policy(self, state: jnp.ndarray) -> jnp.ndarray:
        probability_distribution = self._actor.calculate_actions(state)[0]
        return probability_distribution

    def save(self):
        self._actor.save()
        self._critic.save()

    def load(self):
        self._actor.load()
        self._critic.load()
