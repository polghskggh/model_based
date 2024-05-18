import distrax
import jax
import jax.numpy as jnp
import rlax
from jax import jit

from src.agent.actorcritic.actor import Actor
from src.agent.actorcritic.critic import Critic
from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.enviroment import Shape
from src.models.agent.actorcritic import ActorCritic
from src.models.modelwrapper import ModelWrapper
from src.pod.storage import store, PPOStorage
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key


class PPOStrategy(StrategyInterface):
    def __init__(self):
        self._actor, self._critic = Actor(), Critic()
        self._actor_critic = ModelWrapper(ActorCritic(Shape()[0], (Shape()[1], 1)), "actor_critic")
        self.batch_shape = (Args().args.trajectory_length,
                            Args().args.num_agents)

        self._trajectory_storage = self._init_storage()
        self._iteration: int = 0
        self.old_log_probs = None

    def _init_storage(self):
        return PPOStorage(observations=jnp.zeros((self.batch_shape) + Shape()[0]),
                          rewards=jnp.zeros(self.batch_shape),
                          actions=jnp.zeros(self.batch_shape),
                          log_probs=jnp.zeros(self.batch_shape),
                          dones=jnp.zeros(self.batch_shape),
                          values=jnp.zeros(self.batch_shape))

    def timestep_callback(self, old_state: jax.Array, selected_action: jax.Array, reward: jax.Array,
                          new_state: jax.Array, done: jax.Array):
        self._trajectory_storage = store(self._trajectory_storage, self._iteration, observations=old_state,
                                         actions=selected_action, rewards=reward, dones=done,
                                         log_probs=self.old_log_probs)
        self._iteration += 1

        if self._iteration == Args().args.trajectory_length:
            self.update(new_state)
            self._iteration = 0

    def update(self, term_state):
        _, term_value = self._actor_critic.forward(term_state)
        advantages = self.generalized_advantage_estimation(self._trajectory_storage.values,
                                                            self._trajectory_storage.rewards,
                                                            self._trajectory_storage.dones, term_value,
                                                            Args().args.discount_factor, Args().args.gae_lamda)

        log_probs = self._trajectory_storage.log_probs
        batch_observations = self._trajectory_storage.observations.reshape(-1, Shape()[0])
        batch_actions = self._trajectory_storage.actions.reshape(-1)
        batch_log_probs = self._trajectory_storage.log_probs.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_returns = (advantages + self._trajectory_storage.values).reshape(-1)
        batch_size = Args().args.batch_size

        for start_idx in range(0, self.batch_shape[0], batch_size):
            batch_slice = slice(start_idx, start_idx + batch_size)
            batch_observations = batch_observations[batch_slice]
            actor_grads = self._actor.calculate_grads(batch_observations,
                                                      batch_actions[batch_slice],
                                                      bnlog_probs[batch_slice],
                                                      advantages[batch_slice])

            critic_grads = self._critic.calculate_grads(batch_observations,
                                                        returns[batch_slice])

            self._actor.update(actor_grads)
            self._critic.update(critic_grads)

    def select_action(self, state: jnp.ndarray) -> int:
        policy = self._actor.policy(state)

        action = policy.sample(seed=Key().key(1))
        self.old_log_probs = policy.log_prob(action)
        return action

    @staticmethod
    @jit
    def generalized_advantage_estimation(values, rewards, dones, lambda_, discount_factor, term_value):
        def fold_left(accumulator, rest):
            td_error, discount = rest
            accumulator = td_error + discount * lambda_ * accumulator
            return accumulator, accumulator

        discounts = jnp.where(dones, 0, discount_factor)
        td_errors = rewards + discounts * jnp.append(values[1:], term_value) - values

        _, advantages = jax.lax.scan(fold_left, jnp.zeros(td_errors.shape[1]), (td_errors, discounts),
                                         reverse=True)
        return advantages

    @staticmethod
    def ppo_grad(train_state, states: jax.Array, advantage: jax.Array, action_index: jax.Array,
                 old_log_probs: jax.Array, epsilon: float, regularization: float, rng: dict):
        action_logits = jit(model.apply)(params, states, rngs=rng)
        policy = distrax.Categorical(action_logits)
        log_probs = policy.log_prob(action_index)

        log_ratio = log_probs - old_log_probs
        ratio = jnp.exp(log_ratio)

        approx_kl = ((ratio - 1) - log_ratio).mean()

        loss = rlax.clipped_surrogate_pg_loss(ratio, advantage, epsilon)
        entropy_loss = policy.entropy().mean()

        return (loss - regularization * entropy_loss,
                {"policy_loss": loss, "entropy_loss": entropy_loss, "kl_divergence": approx_kl})

    def save(self):
        self._actor.save()
        self._critic.save()

    def load(self):
        self._actor.load()
        self._critic.load()
