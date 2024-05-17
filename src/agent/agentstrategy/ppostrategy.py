import jax
import jax.numpy as jnp

from src.agent.actorcritic.actor import Actor
from src.agent.actorcritic.critic import Critic
from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.enviroment import Shape
from src.pod.storage import store, PPOStorage
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key


class PPOStrategy(StrategyInterface):
    def __init__(self):
        self._actor, self._critic = Actor(), Critic()

        self.batch_shape = (Args().args.trajectory_length, Args().args.num_agents)

        self._trajectory_storage = self._init_storage()
        self._iteration: int = 0
        self.old_log_probs = None

    def _init_storage(self):
        return PPOStorage(observations=jnp.zeros((self.batch_shape[0] + 1, self.batch_shape[1]) + Shape()[0]),
                          rewards=jnp.zeros(self.batch_shape),
                          actions=jnp.zeros(self.batch_shape),
                          log_probs=jnp.zeros(self.batch_shape),
                          dones=jnp.zeros(self.batch_shape))

    def update(self, old_state: jax.Array, selected_action: jax.Array, reward: jax.Array,
               new_state: jax.Array, done: jax.Array):
        self._trajectory_storage = store(self._trajectory_storage, self._iteration, observations=old_state,
                                         actions=selected_action, rewards=reward, dones=done,
                                         log_probs=self.old_log_probs)
        self._iteration += 1

        if self._iteration != self.batch_shape[0]:
            return

        self._trajectory_storage = store(self._trajectory_storage, self._iteration, observations=new_state)
        observations = self._trajectory_storage.observations
        actions = self._trajectory_storage.actions
        dones = self._trajectory_storage.dones
        log_probs = self._trajectory_storage.log_probs

        advantages, returns = self._critic.provide_feedback(observations, self._trajectory_storage.rewards, dones)

        observations = observations[:-1].reshape(-1, *Shape()[0])
        actions, dones, log_probs = actions.reshape(-1), dones.reshape(-1), log_probs.reshape(-1)

        batch_size = Args().args.batch_size
        for epoch in range(Args().args.num_epochs):
            for start_idx in range(0, self.batch_shape[0], batch_size):
                batch_slice = slice(start_idx, start_idx + batch_size)
                batch_observations = observations[batch_slice]
                actor_grads = self._actor.calculate_grads(batch_observations,
                                                          actions[batch_slice],
                                                          log_probs[batch_slice],
                                                          advantages[batch_slice])

                critic_grads = self._critic.calculate_grads(batch_observations,
                                                            returns[batch_slice])

                self._actor.update(actor_grads)
                self._critic.update(critic_grads)
        self._iteration = 0

    def select_action(self, state: jnp.ndarray) -> int:
        policy = self._actor.policy(state)

        action = policy.sample(seed=Key().key(1))
        self.old_log_probs = policy.log_prob(action)
        return action

    def save(self):
        self._actor.save()
        self._critic.save()

    def load(self):
        self._actor.load()
        self._critic.load()
