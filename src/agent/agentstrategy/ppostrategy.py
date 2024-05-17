import jax
import jax.numpy as jnp
from jax import lax

from src.agent.actor.actor import Actor
from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.agent.critic.ppocritic import PPOCritic
from src.enviroment import Shape
from src.pod.montecarlostorage import MonteCarloStorage
from src.pod.storage import store, Storage
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key


class PPOStrategy(StrategyInterface):
    def __init__(self):
        self._actor, self._critic = Actor(), PPOCritic()

        self.batch_shape = (Args().args.trajectory_length, Args().args.num_agents)

        self._trajectory_storage = self._init_storage()
        self._iteration: int = 0
        self.old_log_odds = None

    def _init_storage(self):
        return Storage(observations=jnp.zeros((self.batch_shape[0] + 1, self.batch_shape[1]) + Shape()[0]),
                       rewards=jnp.zeros(self.batch_shape),
                       actions=jnp.zeros(self.batch_shape),
                       logprobs=jnp.zeros(self.batch_shape),
                       dones=jnp.zeros(self.batch_shape))

    def update(self, old_state: jax.Array, selected_action: jax.Array, reward: jax.Array,
               new_state: jax.Array, done: jax.Array):
        self._trajectory_storage = store(self._trajectory_storage, self._iteration, observations=old_state,
                                         actions=selected_action, rewards=reward, dones=done, logprobs=self.old_log_odds)
        self._iteration += 1

        if self._iteration != self.batch_shape[0]:
            return

        self._trajectory_storage = store(self._trajectory_storage, self._iteration, observations=new_state)
        observations = self._trajectory_storage.observations
        actions = self._trajectory_storage.actions
        dones = self._trajectory_storage.dones
        logprobs = self._trajectory_storage.logprobs

        advantages, returns = self._critic.provide_feedback(observations, self._trajectory_storage.rewards, dones)

        observations = observations[:-1].reshape(-1, *Shape()[0])
        actions, dones, logprobs = actions.reshape(-1), dones.reshape(-1), logprobs.reshape(-1)

        batch_size = Args().args.batch_size
        for epoch in range(Args().args.num_epochs):
            for start_idx in range(0, self.batch_shape[0], batch_size):
                batch_slice = slice(start_idx, start_idx + batch_size)
                actor_grads = self._actor.calculate_grads(observations[batch_slice],
                                                          actions[batch_slice],
                                                          logprobs[batch_slice],
                                                          advantages[batch_slice])

                critic_grads = self._critic.calculate_grads(self._trajectory_storage.observations[batch_slice],
                                                            returns[batch_slice])

                self._actor.update(actor_grads)
                self._critic.update(critic_grads)

        self._iteration = 0

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
