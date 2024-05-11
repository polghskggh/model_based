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
from src.utils.rebatch import rebatch


class PPOStrategy(StrategyInterface):
    def __init__(self):
        self._actor, self._critic = PPOActor(ActorAtari(*Shape())), PPOCritic(StateValueAtariNN(Shape()[0], 1))
        self._trajectory_storage = MonteCarloStorage()
        self._iteration: int = 0
        self._workers = 1

    def is_update_time(self):
        return self._iteration != 0 and self._iteration % hyperparameters["ppo"]["number_of_trajectories"] == 0

    def update(self, old_state: jnp.ndarray, selected_action: int, reward: float, new_state: jnp.ndarray, done: bool):
        self._trajectory_storage.add_transition(old_state, selected_action, reward)

        if done:
            self._iteration += 1
            self._trajectory_storage.end_episode(new_state)

        # Only update every number_of_trajectories
        if not self.is_update_time():
            return

        states, actions, rewards = self._trajectory_storage.data()

        rewards_to_go = self._critic.calculate_rewards_to_go(rewards, states)
        advantage = self._critic.provide_feedback(states, rewards)

        # remove end state
        truncated_states = lax.slice_in_dim(states, start_index=0, limit_index=states.shape[1] - 1, axis=1)

        batch_size = min(hyperparameters['ppo']['batch_size'], states.shape[0] + states.shape[1])
        trunc_states, advantage, actions, rewards_to_go = rebatch(batch_size, truncated_states,
                                                                  advantage, actions, rewards_to_go)

        for trunc_state, adv, action, reward in zip(trunc_states, advantage, actions, rewards_to_go):
            actor_grads = self._actor.calculate_grads(trunc_state, adv, action)
            critic_grads = self._critic.calculate_grads(trunc_state, reward)
            self._actor.update(actor_grads)
            self._critic.update(critic_grads)

        self._trajectory_storage.reset()
        self._iteration = 0

    def action_policy(self, state: jnp.ndarray) -> jnp.ndarray:
        probability_distribution = jnp.squeeze(self._actor.calculate_actions(state))
        return probability_distribution

    def run_parallel(self, n_workers: int):
        self._trajectory_storage = MonteCarloStorage(n_workers)


    def save(self):
        self._actor.save()
        self._critic.save()

    def load(self):
        self._actor.load()
        self._critic.load()
