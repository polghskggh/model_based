import jax
import jax.numpy as jnp
from jax import lax

from src.agent.actor.ppoactor import PPOActor
from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.agent.critic.ppocritic import PPOCritic
from src.enviroment.shape import Shape
from src.models.actorcritic.actoratari import ActorAtari
from src.models.actorcritic.atarinn import StateValueAtariNN
from src.pod.hyperparameters import hyperparameters
from src.pod.montecarlostorage import MonteCarloStorage
from src.pod.trajectorystorage import TrajectoryStorage


class PPOStrategy(StrategyInterface):
    def __init__(self):
        self._actor, self._critic = PPOActor(ActorAtari(*Shape())), PPOCritic(StateValueAtariNN(Shape()[0], 1))
        self._trajectory_storage = MonteCarloStorage()
        self._number_of_trajectories: int = hyperparameters['ppo']['number_of_trajectories']
        self._iteration: int = 0

    def _batch_update(self, training_sample: list[jax.Array]):
        states = training_sample[0]
        action = training_sample[1]
        rewards = training_sample[2]

        advantage = self._critic.provide_feedback(states, rewards)

        # remove end state
        truncated_states = lax.slice_in_dim(states, start_index=0, limit_index=states.shape[1] - 1, axis=1)
        actor_grads = self._actor.calculate_grads(truncated_states, advantage, action)
        critic_grads = self._critic.calculate_grads(states, rewards)
        return actor_grads, critic_grads

    def is_update_time(self):
        return self._iteration != 0 and self._iteration % self._number_of_trajectories == 0

    def update(self, old_state: jnp.ndarray, selected_action: int, reward: float, new_state: jnp.ndarray, done: bool):
        self._trajectory_storage.add_transition(old_state, selected_action, reward)

        if done:
            self._iteration += 1
            self._trajectory_storage.end_episode(new_state)

        # Only update every number_of_trajectories
        if not self.is_update_time():
            return

        transitions = self._trajectory_storage.data()

        actor_grads, critic_grads = self._batch_update(transitions)

        self._actor.update(actor_grads)
        self._critic.update(critic_grads)

        self._trajectory_storage.reset()
        self._iteration = 0

    def action_policy(self, state: jnp.ndarray) -> jnp.ndarray:
        probability_distribution = self._actor.calculate_actions(state)[0]
        return probability_distribution

    def save(self):
        self._actor.save()
        self._critic.save()

    def load(self):
        self._actor.load()
        self._critic.load()
