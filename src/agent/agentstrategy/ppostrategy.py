import distrax
import jax
import jax.numpy as jnp
import optax
import rlax
from jax import jit, value_and_grad

from altmodel import Network
from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.enviroment import Shape
from src.models.agent.actorcritic import ActorCritic
from src.models.modelwrapper import ModelWrapper
from src.pod.storage import store, PPOStorage
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key
from src.singletons.writer import log


class PPOStrategy(StrategyInterface):
    def __init__(self):
        self._actor_critic = ModelWrapper(ActorCritic(Shape()[0], (Shape()[1], 1)), "actor_critic")
        #self._actor_critic = ModelWrapper(Network(Shape()[0]), "actor_critic")
        self.batch_shape = (Args().args.trajectory_length,
                            Args().args.num_agents)

        self._trajectory_storage = self._init_storage()
        self._iteration: int = 0

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
                                         actions=selected_action, rewards=reward, dones=done)
        self._iteration += 1

        if self._iteration == Args().args.trajectory_length:
            self.update(new_state)
            self._iteration = 0

    def update(self, term_state):
        _, term_value = self._actor_critic.forward(term_state)
        advantages = self.generalized_advantage_estimation(self._trajectory_storage.values,
                                                            self._trajectory_storage.rewards,
                                                            self._trajectory_storage.dones, term_value.squeeze(),
                                                            Args().args.discount_factor, Args().args.gae_lambda)
        batch_observations = self._trajectory_storage.observations.reshape(-1, *Shape()[0])
        batch_actions = self._trajectory_storage.actions.reshape(-1)
        batch_log_probs = self._trajectory_storage.log_probs.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_returns = (advantages + self._trajectory_storage.values).reshape(-1)
        batch_size = Args().args.batch_size

        grad_fn = jit(value_and_grad(PPOStrategy.ppo_loss, 1, has_aux=True))
        for _ in range(Args().args.num_epochs):
            for start_idx in range(0, self.batch_shape[0], batch_size):
                batch_slice = slice(start_idx, start_idx + batch_size)
                (loss, aux), grads = grad_fn(self._actor_critic.state,
                                             self._actor_critic.params,
                                             batch_observations[batch_slice],
                                             batch_actions[batch_slice],
                                             batch_log_probs[batch_slice],
                                             batch_advantages[batch_slice],
                                             batch_returns[batch_slice])
                log(aux)
                self._actor_critic.apply_grads(grads)

    @staticmethod
    def ppo_loss(state, params, states: jax.Array, actions, old_log_probs: jax.Array, advantages: jax.Array,
                 returns: jax.Array, args=Args().args):
        action_logits, values = state.apply_fn(params, states)
        policy = distrax.Categorical(action_logits)
        log_probs = policy.log_prob(actions)
        log_ratio = log_probs - old_log_probs
        ratio = jnp.exp(log_ratio)

        approx_kl = ((ratio - 1) - log_ratio).mean()

        loss = rlax.clipped_surrogate_pg_loss(ratio, advantages, args.clip_threshold)
        entropy_loss = policy.entropy().mean()

        value_loss = jnp.mean(optax.squared_error(values.squeeze(), returns))

        return (loss - args.regularization * entropy_loss + args.value_weight * value_loss,
                {"policy_loss": loss, "entropy_loss": entropy_loss, "kl_divergence": approx_kl,
                 "value_loss": value_loss})

    def select_action(self, states: jnp.ndarray, store: bool) -> int:
        logits, value_estimate = self._actor_critic.forward(states)
        value_estimate = value_estimate.squeeze()
        policy = distrax.Categorical(logits)
        action = policy.sample(seed=Key().key(1))
        if store:
            self._trajectory_storage = store(self._trajectory_storage, self._iteration,
                                             log_probs=policy.log_prob(action), values=value_estimate)
        return action

    @staticmethod
    @jit
    def generalized_advantage_estimation(values, rewards, dones, term_value, discount_factor, lambda_):
        def fold_left(accumulator, rest):
            td_error, discount = rest
            accumulator = td_error + discount * lambda_ * accumulator
            return accumulator, accumulator

        discounts = jnp.where(dones, 0, discount_factor)
        td_errors = rewards + discounts * jnp.append(values[1:], jnp.expand_dims(term_value, 0), axis=0) - values

        _, advantages = jax.lax.scan(fold_left, jnp.zeros(td_errors.shape[1]), (td_errors, discounts), reverse=True)
        return advantages

    def save(self):
        pass

    def load(self):
        pass
