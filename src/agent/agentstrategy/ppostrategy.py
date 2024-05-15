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
        self.old_log_odds = None

    def is_update_time(self):
        return self._iteration != 0 and self._iteration % hyperparameters["ppo"]["number_of_trajectories"] == 0

    def update(self, old_state: jnp.ndarray, selected_action: int, reward: float,
               new_state: jnp.ndarray, term: bool, trunc: bool):
        self._trajectory_storage.add_transition(old_state, selected_action, reward, term, self.old_log_odds)

        if term or trunc:
            self._iteration += 1
            self._trajectory_storage.end_episode(new_state)

        # Only update every number_of_trajectories
        if not self.is_update_time():
            return

        states, actions, rewards, dones, old_log_odds = self._trajectory_storage.data()
        returns = self._critic.calculate_rewards_to_go(rewards, states, dones)
        advantages = self._critic.provide_feedback(states, rewards, dones)
        print(returns)
        print(advantages)

        # remove end state
        truncated_states = lax.slice_in_dim(states, start_index=0, limit_index=states.shape[1] - 1, axis=1)
        batch_size = min(hyperparameters['ppo']['batch_size'], truncated_states.shape[0] + truncated_states.shape[1])
        trunc_states, advantage, actions, returns, old_log_odds = rebatch(batch_size, truncated_states,
                                                                          advantages, actions, returns, old_log_odds)
        for trunc_state, adv, action, ret, old_log_odd in zip(trunc_states, advantage, actions, returns, old_log_odds):
            actor_grads = self._actor.calculate_grads(trunc_state, adv, action, old_log_odd)
            critic_grads = self._critic.calculate_grads(trunc_state, ret)

            self._actor.update(actor_grads)
            self._critic.update(critic_grads)

        self._trajectory_storage.reset()
        self._iteration = 0

    def select_action(self, state: jnp.ndarray) -> int:
        policy, logits = self._actor.policy(state)

        policy = jnp.squeeze(policy)
        self.old_log_odds = jnp.squeeze(logits)

        sample_fun = self.__sample_from_distribution
        if len(policy.shape) > 1:
            sample_fun = vmap(sample_fun)
        print("following policy", policy)
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
