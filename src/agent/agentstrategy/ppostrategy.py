import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from flax.linen import softmax

from src.agent.actor.actor import Actor
from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.agent.critic.ppocritic import PPOCritic
from src.enviroment.shape import Shape
from src.models.actorcritic.actoratari import ActorAtari
from src.models.actorcritic.atarinn import StateValueAtariNN
from src.pod.hyperparameters import hyperparameters
from src.pod.montecarlostorage import MonteCarloStorage
from src.utils.modelhelperfuns import sum_dicts
from src.utils.rebatch import rebatch


class PPOStrategy(StrategyInterface):
    def __init__(self):
        self._actor, self._critic = Actor(), PPOCritic()

        self._key = jr.PRNGKey(hyperparameters["rng"]["action"])
        self._trajectory_storage = MonteCarloStorage()
        self._iteration: int = 0
        self._workers = 1
        self._last_policy = None

    def is_update_time(self):
        return self._iteration != 0 and self._iteration % hyperparameters["ppo"]["number_of_trajectories"] == 0

    def update(self, old_state: jnp.ndarray, selected_action: int, reward: float, new_state: jnp.ndarray, done: bool):
        self._trajectory_storage.add_transition(old_state, selected_action, reward, self._last_policy)

        if done:
            self._iteration += 1
            self._trajectory_storage.end_episode(new_state)

        # Only update every number_of_trajectories
        if not self.is_update_time():
            return

        states, actions, rewards, old_policy = self._trajectory_storage.data()
        returns = self._critic.calculate_rewards_to_go(rewards, states)
        advantages = self._critic.provide_feedback(states, rewards)

        # remove end state
        truncated_states = lax.slice_in_dim(states, start_index=0, limit_index=states.shape[1] - 1, axis=1)
        batch_size = min(hyperparameters['ppo']['batch_size'], truncated_states.shape[0] + truncated_states.shape[1])
        trunc_states, advantage, actions, returns, old_policy = rebatch(batch_size, truncated_states,
                                                            advantages, actions, returns, old_policy)
        for trunc_state, adv, action, ret, old_p in zip(trunc_states, advantage, actions, returns, old_policy):
            actor_grads = self._actor.calculate_grads(trunc_state, adv, action, old_p)
            critic_grads = self._critic.calculate_grads(trunc_state, ret)

            self._actor.update(actor_grads)
            self._critic.update(critic_grads)

        self._trajectory_storage.reset()
        self._iteration = 0

    def action_policy(self, state: jnp.ndarray) -> jnp.ndarray:
        probability_distribution = jnp.squeeze(self._actor.policy(state))
        self._last_policy = probability_distribution
        return probability_distribution

    def select_action(self, state: jnp.ndarray) -> int:
        policy = self.action_policy(state)
        sample_fun = self.__sample_from_distribution
        if len(policy.shape) > 1:
            sample_fun = vmap(sample_fun)

        return sample_fun(policy)

    def run_parallel(self, n_workers: int):
        self._trajectory_storage = MonteCarloStorage(n_workers)

    def save(self):
        self._actor.save()
        self._critic.save()

    def load(self):
        self._actor.load()
        self._critic.load()

    def __sample_from_distribution(self, distribution: jax.Array) -> int:
        self._key, subkey = jr.split(self._key)
        return int(jr.choice(subkey, distribution.shape[0], p=distribution))
