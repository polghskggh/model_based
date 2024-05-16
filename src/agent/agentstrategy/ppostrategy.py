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
from src.pod.montecarlostorage import MonteCarloStorage
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key
from src.utils.modelhelperfuns import sum_dicts
from src.utils.rebatch import rebatch


class PPOStrategy(StrategyInterface):
    def __init__(self):
        self._actor, self._critic = Actor(), PPOCritic()

        self._trajectory_storage = MonteCarloStorage()
        self._iteration: int = 0
        self.old_log_odds = None

    def is_update_time(self):
        return

    def update(self, old_state: jax.Array, selected_action: jax.Array, reward: jax.Array,
               new_state: jax.Array, done: jax.Array):
        self._trajectory_storage.add_transition(old_state, selected_action, reward, done, self.old_log_odds)

        if not all(done):
            return

        self._trajectory_storage.end_episode(new_state)

        states, actions, rewards, dones, old_log_odds = self._trajectory_storage.data()
        advantages, returns = self._critic.provide_feedback(states, rewards, dones)

        # remove end state
        truncated_states = lax.slice_in_dim(states, start_index=0, limit_index=states.shape[1] - 1, axis=1)
        batch_size = min(Args().args.batch_size, truncated_states.shape[0] + truncated_states.shape[1])
        trunc_states, advantage, actions, returns, old_log_odds = rebatch(batch_size, truncated_states,
                                                                          advantages, actions, returns, old_log_odds)

        for trunc_state, adv, action, ret, old_log_odd in zip(trunc_states, advantage, actions, returns, old_log_odds):
            actor_grads = self._actor.calculate_grads(trunc_state, adv, action, old_log_odd)
            critic_grads = self._critic.calculate_grads(trunc_state, ret)

            self._actor.update(actor_grads)
            self._critic.update(critic_grads)

        self._trajectory_storage.reset()

    def select_action(self, state: jnp.ndarray) -> int:
        policy = self._actor.policy(state)

        action = policy.sample(seed=Key().key(1))
        self.old_log_odds = policy.log_prob(action)
        return action

    def run_parallel(self, n_workers: int):
        self._trajectory_storage = MonteCarloStorage(n_workers)

    def save(self):
        self._actor.save()
        self._critic.save()

    def load(self):
        self._actor.load()
        self._critic.load()
