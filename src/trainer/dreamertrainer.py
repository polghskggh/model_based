import distrax
import jax
import jax.numpy as jnp
from jax import value_and_grad, lax

from src.enviroment import Shape
from src.models.lossfuns import reward_loss_fn, image_loss_fn, softmax_loss
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key
from src.singletons.writer import log
from src.trainer.trainer import Trainer


class DreamerTrainer(Trainer):
    def __init__(self, models: dict):
        self.belief_size = Args().args.belief_size
        self.state_size = Args().args.state_size
        self.models = models

    def apply_grads(self, grads):
        for key in grads.keys():
            self.models[key].apply_grads(grads[key])

    def train_step(self, initial_belief, initial_state, observations, actions, rewards, dones):
        rng = {"normal": Key().key()}
        last_belief, last_state = initial_belief, initial_state
        keys_to_select = ['representation', 'observation', 'reward', 'encoder']

        if Args().args.predict_dones:
            keys_to_select.append('dones')

        params = {key: self.models[key].params for key in keys_to_select}
        models = {key: self.models[key].model for key in keys_to_select}

        batch_size = Args().args.batch_size
        for start_idx in range(0, observations.shape[0], batch_size):
            batch_slice = slice(start_idx, start_idx + batch_size)
            data = (observations[batch_slice], actions[batch_slice], rewards[batch_slice], dones[batch_slice],
                    last_state, last_belief)
            (loss, aux), grads = value_and_grad(self.loss_fun, 1, True)(models, params, data, rng)
            self.apply_grads(grads)
            last_belief, last_state = aux["data"]
            log(aux["info"])

        new_params = {"params": self.models["representation"].params["params"]["transition_model"]}
        self.models["transition"].params = new_params
        return self.models

    @staticmethod
    def loss_fun(models: dict, params: dict, data: tuple, rng: dict):
        observations, actions, rewards, dones, state, belief = data
        batch_size = Args().args.batch_size

        beliefs = jnp.zeros((observations.shape[0], Args().args.num_envs, Args().args.belief_size))
        state_shape = (observations.shape[0], Args().args.num_envs, Args().args.state_size)
        prior_means = jnp.zeros(state_shape)
        prior_std_devs = jnp.zeros(state_shape)
        states = jnp.zeros(state_shape)
        posterior_means = jnp.zeros(state_shape)
        posterior_std_devs = jnp.zeros(state_shape)
        old_shape = observations.shape

        observations = observations.reshape(-1, *observations.shape[2:])
        encoded_observations = jnp.zeros((observations.shape[0], ) + Args().args.bottleneck_dims)
        key = "encoder"
        for start_idx in range(0, observations.shape[0], batch_size):
            batch_slice = slice(start_idx, start_idx + batch_size)
            encoded_batch, _ = models[key].apply(params[key], observations[batch_slice], rngs=rng)
            encoded_observations = encoded_observations.at[batch_slice].set(encoded_batch)

        encoded_observations = encoded_observations.reshape(old_shape[:2] + Args().args.bottleneck_dims)

        key = "representation"
        for idx in range(len(states)):
            output = models[key].apply(params[key], state, actions[idx], belief, encoded_observations[idx], rngs=rng)
            beliefs = beliefs.at[idx].set(output[0])
            states = states.at[idx].set(output[1])
            prior_means = prior_means.at[idx].set(output[2])
            prior_std_devs = prior_std_devs.at[idx].set(output[3])
            posterior_means = posterior_means.at[idx].set(output[4])
            posterior_std_devs = posterior_std_devs.at[idx].set(output[5])

            belief = beliefs[idx]
            state = states[idx]

        beliefs = beliefs.reshape(-1, Args().args.belief_size)
        prior_means = prior_means.reshape(-1)
        prior_std_devs = prior_std_devs.reshape(-1)
        states = states.reshape(-1, Args().args.state_size)
        posterior_means = posterior_means.reshape(-1)
        posterior_std_devs = posterior_std_devs.reshape(-1)

        # n_channels = Shape()[0][2] // Args().args.frame_stack
        # observations = jax.lax.slice_in_dim(observations, (Args().args.frame_stack - 1) * n_channels,
        #                                      None, axis=-1)
        rewards = rewards.reshape(-1)
        dones = dones.reshape(-1)

        observation_loss, reward_loss, dones_loss = 0, 0, 0
        num_batches = 0

        for start_idx in range(0, observations.shape[0], Args().args.batch_size):
            num_batches += 1
            batch_slice = slice(start_idx, start_idx + batch_size)
            key = "observation"
            pixels = models[key].apply(params[key], beliefs[batch_slice], states[batch_slice])

            observation_loss += image_loss_fn(pixels, observations[batch_slice])

            key = "reward"
            reward_logits = models[key].apply(params[key], beliefs[batch_slice], states[batch_slice])
            reward_loss += reward_loss_fn(reward_logits, rewards[batch_slice])

            if Args().args.predict_dones:
                key = "dones"
                dones_logits = models[key].apply(params[key], beliefs[batch_slice], states[batch_slice])
                dones_loss += jnp.mean(softmax_loss(dones_logits, dones[batch_slice]))

        num_batches = jnp.minimum(num_batches, 1)
        observation_loss /= num_batches
        reward_loss /= num_batches
        dones_loss /= num_batches

        distribution = distrax.MultivariateNormalDiag(prior_means, prior_std_devs)
        kl_loss = distribution.kl_divergence(distrax.MultivariateNormalDiag(posterior_means, posterior_std_devs))
        kl_loss /= jnp.minimum(posterior_std_devs.shape[0], 1)

        alpha, beta, gamma = Args().args.loss_weights
        return (alpha * observation_loss + beta * reward_loss + beta * dones_loss + gamma * kl_loss,
                {
                    "info": {"observation_loss": observation_loss, "reward_loss": reward_loss, "kl_loss": kl_loss,
                             "dones_loss": dones_loss},
                    "data": (belief, state)
                })
