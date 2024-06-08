import distrax
import jax
import jax.numpy as jnp
import optax
import rlax
from jax import jit, value_and_grad

from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.enviroment import Shape
from src.models.agent.actorcritic import ActorCritic, ActorCriticDreamer
from src.models.modelwrapper import ModelWrapper
from src.pod.storage import store, PPOStorage
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key
from src.singletons.writer import log


class PPOStrategy(StrategyInterface):
    def __init__(self):
        if Args().args.algorithm == "dreamer":
            self.state_shape = (Args().args.state_size, )
            model = ActorCriticDreamer(self.state_shape, (Shape()[1], 1))
        else:
            self.state_shape = Shape()[0]
            model = ActorCritic(Shape()[0], (Shape()[1], 1))

        self._actor_critic = ModelWrapper(model, "actor_critic")

        if Args().args.algorithm == "dreamer" or Args().args.algorithm == "simple":
            self.trajectory_length = Args().args.sim_trajectory_length
        else:
            self.trajectory_length = Args().args.trajectory_length

        self._batch_shape = (self.trajectory_length, Args().args.num_agents)

        self._trajectory_storage = self._init_storage()
        self._iteration: int = 0

    def _init_storage(self):
        return PPOStorage(observations=jnp.zeros((self._batch_shape) + self.state_shape),
                          rewards=jnp.zeros(self._batch_shape),
                          actions=jnp.zeros(self._batch_shape),
                          log_probs=jnp.zeros(self._batch_shape),
                          dones=jnp.zeros(self._batch_shape),
                          values=jnp.zeros(self._batch_shape))

    def timestep_callback(self, old_state: jax.Array, selected_action: jax.Array, reward: jax.Array,
                          new_state: jax.Array, done: jax.Array):
        self._trajectory_storage = store(self._trajectory_storage, self._iteration, observations=old_state,
                                         actions=selected_action, rewards=reward, dones=done)
        self._iteration += 1

        if self._iteration == self.trajectory_length:
            self.update(new_state)
            self._iteration = 0

    def update(self, last_state):
        _, last_value = self._actor_critic.forward(last_state)
        advantages = self.generalized_advantage_estimation(self._trajectory_storage.values,
                                                            self._trajectory_storage.rewards,
                                                            self._trajectory_storage.dones, last_value.squeeze(),
                                                            Args().args.discount_factor, Args().args.gae_lambda)
        batch_observations = self._trajectory_storage.observations.reshape(-1, *self.state_shape)
        batch_actions = self._trajectory_storage.actions.reshape(-1)
        batch_log_probs = self._trajectory_storage.log_probs.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_values = self._trajectory_storage.values.reshape(-1)
        batch_returns = batch_advantages + batch_values
        batch_size = Args().args.batch_size

        grad_fn = jit(value_and_grad(PPOStrategy.ppo_loss, 1, has_aux=True))
        for _ in range(Args().args.num_epochs):
            for start_idx in range(0, self._batch_shape[0], batch_size):
                batch_slice = slice(start_idx, start_idx + batch_size)
                (loss, aux), grads = grad_fn(self._actor_critic.state,
                                             self._actor_critic.params,
                                             batch_observations[batch_slice],
                                             batch_actions[batch_slice],
                                             batch_log_probs[batch_slice],
                                             batch_values[batch_slice],
                                             batch_advantages[batch_slice],
                                             batch_returns[batch_slice])
                log(aux)
                self._actor_critic.apply_grads(grads)

    @staticmethod
    def ppo_loss(state, params, states: jax.Array, actions, old_log_probs: jax.Array, values: jax.Array,
                 advantages: jax.Array, returns: jax.Array, args=Args().args):
        action_logits, new_values = state.apply_fn(params, states)
        new_values = new_values.squeeze()
        policy = distrax.Categorical(action_logits)
        log_probs = policy.log_prob(actions)
        log_ratio = log_probs - old_log_probs
        ratio = jnp.exp(log_ratio)

        approx_kl = ((ratio - 1) - log_ratio).mean()

        policy_loss = rlax.clipped_surrogate_pg_loss(ratio, advantages, args.clip_threshold)
        entropy_loss = jnp.mean(policy.entropy())

        value_loss_unclipped = optax.squared_error(new_values, returns)
        value_clipped = values + jnp.clip(new_values - values, -args.clip_threshold, args.clip_threshold)
        value_loss_clipped = optax.squared_error(value_clipped, returns)
        value_loss_max = jnp.maximum(value_loss_clipped, value_loss_unclipped)
        value_loss = 0.5 * jnp.mean(value_loss_max)

        return (policy_loss - args.regularization * entropy_loss + args.value_weight * value_loss,
                {"policy_loss": policy_loss, "entropy_loss": entropy_loss, "kl_divergence": approx_kl,
                 "value_loss": value_loss})

    def select_action(self, states: jnp.ndarray, store_trajectories: bool) -> int:
        logits, value_estimate = self._actor_critic.forward(states)
        logits = jnp.squeeze(logits)
        value_estimate = value_estimate.squeeze()
        policy = distrax.Categorical(logits)
        action = policy.sample(seed=Key().key(1))
        print("debug", action, policy.log_prob(action))
        if store_trajectories:
            self._trajectory_storage = store(self._trajectory_storage, self._iteration,
                                             log_probs=policy.log_prob(action), values=value_estimate)
        return action.squeeze()

    @staticmethod
    @jit
    def generalized_advantage_estimation(values, rewards, dones, term_value, discount_factor, lambda_):
        def fold_left(last_gae, rest):
            td_error, discount = rest   
            last_gae = td_error + discount * lambda_ * last_gae
            return last_gae, last_gae

        discounts = jnp.where(dones, 0, discount_factor)
        td_errors = rewards + discounts * jnp.append(values[1:], jnp.expand_dims(term_value, 0), axis=0) - values

        _, advantages = jax.lax.scan(fold_left, jnp.zeros(td_errors.shape[1]), (td_errors, discounts), reverse=True)
        return advantages

    def save(self):
        pass

    def load(self):
        pass
