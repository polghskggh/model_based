import distrax
import jax
import optax
from jax import value_and_grad, vmap

from src.models.lossfuns import mean_squared_error
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key
from src.singletons.writer import Writer, log
from src.trainer.trainer import Trainer
import jax.numpy as jnp

from src.utils.rl import tile_image


class DreamerTrainer(Trainer):
    def __init__(self, representation_model, observation_model, reward_model):
        self.batch_size = Args().args.batch_size
        self.belief_size = Args().args.belief_size
        self.state_size = Args().args.state_size

        self.models = {"representation": representation_model,
                       "observation": observation_model,
                       "reward": reward_model}

    def train_step(self, observations, actions, rewards, params: dict):
        init_belief = jnp.zeros((observations.shape[1], self.belief_size))
        init_state = jnp.zeros((observations.shape[1], self.state_size))

        rng = {"normal": Key().key()}

        batch_size = Args().args.batch_size
        for start_idx in range(0, observations.shape[0], batch_size):
            batch_slice = slice(start_idx, start_idx + batch_size)
            data = (observations[batch_slice], actions[batch_slice], rewards[batch_slice], init_state, init_belief)
            (loss, aux), grads = value_and_grad(self.loss_fun, 1, True)(self.models, params, data, rng)
            init_belief, init_state = aux["data"]
            for key in self.models.keys():
                params[key] = optax.apply_updates(params[key], grads[key])
            log(aux["info"])
        return params

    @staticmethod
    def loss_fun(models: dict, params: dict, data: tuple, rng: dict):
        observations, actions, rewards, state, belief = data

        beliefs = jnp.zeros((observations.shape[0], Args().args.num_envs, Args().args.belief_size))
        state_shape = (observations.shape[0], Args().args.num_envs, Args().args.state_size)
        prior_means = jnp.zeros(state_shape)
        prior_std_devs = jnp.zeros(state_shape)
        states = jnp.zeros(state_shape)
        posterior_means = jnp.zeros(state_shape)
        posterior_std_devs = jnp.zeros(state_shape)

        key = "representation"
        for idx in range(len(states)):
            data = models[key].apply(params[key], state, actions[idx], belief, observations[idx], rngs=rng)
            beliefs.at[idx].set(data[0])
            states.at[idx].set(data[1])
            prior_means.at[idx].set(data[2])
            prior_std_devs.at[idx].set(data[3])
            posterior_means.at[idx].set(data[4])
            posterior_std_devs.at[idx].set(data[5])
            belief = beliefs[idx]
            state = states[idx]

        beliefs = beliefs.reshape(-1, Args().args.belief_size)
        prior_means = prior_means.reshape(-1)
        prior_std_devs = prior_std_devs.reshape(-1)
        states = states.reshape(-1, Args().args.state_size)
        posterior_means = posterior_means.reshape(-1)
        posterior_std_devs = posterior_std_devs.reshape(-1)
        observations = observations.reshape(-1, *observations.shape[2:])
        rewards = rewards.reshape(-1)

        batch_size = Args().args.batch_size
        observation_loss, reward_loss = 0, 0
        num_batches = 0
        observations = vmap(tile_image)(observations)

        for start_idx in range(0, observations.shape[0], Args().args.batch_size):
            num_batches += 1
            batch_slice = slice(start_idx, start_idx + batch_size)
            key = "observation"
            pixels = models[key].apply(params[key], beliefs[batch_slice], states[batch_slice])
            print(pixels.shape, observations[batch_slice].shape)

            observation_loss += jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pixels,
                                                                                         observations[batch_slice]))

            key = "reward"
            reward_logits = models[key].apply(params[key], beliefs[batch_slice], states[batch_slice])
            teacher_rewards = jnp.astype(rewards[batch_slice], jnp.int32)
            reward_loss += jnp.mean(optax.softmax_cross_entropy_with_integer_labels(reward_logits, teacher_rewards))

        observation_loss /= num_batches
        reward_loss /= num_batches

        distribution = distrax.MultivariateNormalDiag(prior_means, prior_std_devs)
        kl_loss = distribution.kl_divergence(distrax.MultivariateNormalDiag(posterior_means, posterior_std_devs))

        alpha, beta, gamma = Args().args.loss_weights
        jax.debug.print("Losses: {obs}, {rew}, {kl}", obs=observation_loss, rew=reward_loss, kl=kl_loss)
        return (alpha * observation_loss + beta * reward_loss + gamma * kl_loss,
                {
                    "info": {"observation_loss": observation_loss, "reward_loss": reward_loss, "kl_loss": kl_loss},
                    "data": (belief, state)
                })
