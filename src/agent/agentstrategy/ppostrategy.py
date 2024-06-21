import distrax
import jax
import jax.numpy as jnp
import optax
import rlax
from jax import jit, value_and_grad

from src.agent.agentstrategy.strategyinterface import StrategyInterface
from src.enviroment import Shape
from src.models.agent.actorcritic import ActorCriticDreamer
from src.models.modelwrapper import ModelWrapper
from src.models.test.testae import ActorCriticNetwork
from src.pod.storage import store, PPOStorage
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key
from src.singletons.writer import log
import jax.random as jr


class PPOStrategy(StrategyInterface):
    def __init__(self):
        if Args().args.algorithm == "dreamer":
            self.state_shape = (Args().args.state_size + Args().args.belief_size, )
            model = ActorCriticDreamer(self.state_shape, (Shape()[1], 1))
        else:
            self.state_shape = Shape()[0]
            model = ActorCriticNetwork(Shape()[0], (Shape()[1], 1))

        self._actor_critic = ModelWrapper(model, "actor_critic")

        if Args().args.algorithm == "dreamer" or Args().args.algorithm == "simple":
            self.trajectory_length = Args().args.sim_trajectory_length
        else:
            self.trajectory_length = Args().args.trajectory_length

        if Args().args.hybrid_learning:
            print("Hybrid Learning")
            self._hybrid_trajectory_storage = self._init_storage((Args().args.trajectory_length, Args().args.num_envs))

        self._trajectory_storage = self._init_storage((self.trajectory_length, Args().args.num_agents))
        self._iteration: int = 0

    def _init_storage(self, batch_shape):
        return PPOStorage(observations=jnp.zeros((batch_shape) + self.state_shape),
                          rewards=jnp.zeros(batch_shape),
                          actions=jnp.zeros(batch_shape),
                          log_probs=jnp.zeros(batch_shape),
                          dones=jnp.zeros(batch_shape),
                          values=jnp.zeros(batch_shape))

    def timestep_callback(self, old_state: jax.Array, reward: jax.Array,
                          new_state: jax.Array, done: jax.Array, store_trajectory: jax.Array):
        if not store_trajectory and not Args().args.hybrid_learning:
            return

        if not store_trajectory:
            storage = self._hybrid_trajectory_storage
            update_time = Args().args.trajectory_length
        else:
            storage = self._trajectory_storage
            update_time = self.trajectory_length

        storage = store(storage, self._iteration, observations=old_state,
                        rewards=reward, dones=done)
        self._iteration += 1

        if self._iteration == update_time:
            self.update(new_state, storage)
            self._iteration = 0

        if not store_trajectory:
            self._hybrid_trajectory_storage = storage
        else:
            self._trajectory_storage = storage

    def update(self, last_state, storage):
        _, last_value = self._actor_critic.forward(last_state)
        print(jnp.unique(storage.actions, return_counts=True))

        print("reward mean", jnp.mean(storage.rewards))
        print("obs means", jnp.mean(storage.observations), jnp.mean(storage.observations[0][0]),
              jnp.mean(storage.observations[0][1]))
        print("dones mean", jnp.mean(storage.dones))
        advantages = self.generalized_advantage_estimation(storage.values, storage.rewards,
                                                           storage.dones, last_value.squeeze(),
                                                           Args().args.discount_factor, Args().args.gae_lambda)
        batch_observations = storage.observations.reshape(-1, *self.state_shape)
        batch_actions = storage.actions.reshape(-1)
        batch_log_probs = storage.log_probs.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_values = storage.values.reshape(-1)
        batch_returns = batch_advantages + batch_values
        batch_size = Args().args.batch_size
        epoch_size = batch_observations.shape[0]

        grad_fn = jit(value_and_grad(PPOStrategy.ppo_loss, 1, has_aux=True))
        for _ in range(Args().args.num_epochs):
            epoch_indices = jr.permutation(Key().key(), epoch_size, independent=True)
            for start_idx in range(0, epoch_size, batch_size):
                end_idx = start_idx + batch_size
                batch_indices = epoch_indices[start_idx:end_idx]
                (loss, aux), grads = grad_fn(self._actor_critic.state,
                                             self._actor_critic.params,
                                             batch_observations[batch_indices],
                                             batch_actions[batch_indices],
                                             batch_log_probs[batch_indices],
                                             batch_values[batch_indices],
                                             batch_advantages[batch_indices],
                                             batch_returns[batch_indices])
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
        value_loss = jnp.mean(optax.squared_error(new_values, returns))

        return (policy_loss - args.regularization * entropy_loss + args.value_weight * value_loss,
                {"policy_loss": policy_loss, "entropy_loss": entropy_loss, "kl_divergence": approx_kl,
                 "value_loss": value_loss})

    def select_action(self, states: jnp.ndarray, store_trajectories: bool) -> int:
        logits, value_estimate = self._actor_critic.forward(states)
        policy = distrax.Categorical(logits=logits.squeeze())

        action = policy.sample(seed=Key().key()).squeeze()

        if store_trajectories:
            self._trajectory_storage = store(self._trajectory_storage, self._iteration,
                                             log_probs=policy.log_prob(action),
                                             actions=action,
                                             values=value_estimate.squeeze())
        elif Args().args.hybrid_learning:
            self._hybrid_trajectory_storage = store(self._hybrid_trajectory_storage, self._iteration,
                                                    log_probs=policy.log_prob(action),
                                                    actions=action,
                                                    values=value_estimate.squeeze())
        return action

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
