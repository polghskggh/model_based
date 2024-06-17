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
        print(grads.keys())
        for key in grads.keys():
            self.models[key].apply_grads(grads[key])

    def train_step(self, initial_belief, initial_state, observations, actions, rewards, dones):
        rng = {"normal": Key().key()}
        last_belief, last_state = initial_belief, initial_state
        keys_to_select = ['representation', 'observation', 'reward']
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

        beliefs = jnp.zeros((observations.shape[0], Args().args.num_envs, Args().args.belief_size))

        state_shape = (observations.shape[0], Args().args.num_envs, Args().args.state_size)
        prior_means = jnp.zeros(state_shape)
        prior_std_devs = jnp.zeros(state_shape)
        states = jnp.zeros(state_shape)
        posterior_means = jnp.zeros(state_shape)
        posterior_std_devs = jnp.zeros(state_shape)

        key = "representation"
        for idx in range(len(states)):
            output = models[key].apply(params[key], state, actions[idx], belief, observations[idx], rngs=rng)
            beliefs = beliefs.at[idx].set(output[0])
            states = states.at[idx].set(output[1])
            prior_means = prior_means.at[idx].set(output[2])
            prior_std_devs = prior_std_devs.at[idx].set(output[3])
            posterior_means = posterior_means.at[idx].set(output[4])
            posterior_std_devs = posterior_std_devs.at[idx].set(output[5])

            condition = jnp.expand_dims(jnp.astype(dones[idx], jnp.bool_), -1)
            belief = jnp.where(condition, jnp.zeros_like(beliefs[idx]), beliefs[idx])
            state = jnp.where(condition, jnp.zeros_like(beliefs[idx]), states[idx])

        beliefs = beliefs.reshape(-1, Args().args.belief_size)
        prior_means = prior_means.reshape(-1)
        prior_std_devs = prior_std_devs.reshape(-1)
        states = states.reshape(-1, Args().args.state_size)
        posterior_means = posterior_means.reshape(-1)
        posterior_std_devs = posterior_std_devs.reshape(-1)
        observations = observations.reshape(-1, *observations.shape[2:])
        n_channels = Shape()[0][2] // Args().args.frame_stack
        observations = lax.slice_in_dim(observations, (Args().args.frame_stack - 1) * n_channels,
                                        None, axis=-1)
        rewards = rewards.reshape(-1)

        batch_size = Args().args.batch_size
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
                dones_loss += softmax_loss(dones_logits, dones[batch_slice])

        observation_loss /= num_batches
        reward_loss /= num_batches
        dones_loss /= num_batches

        distribution = distrax.MultivariateNormalDiag(prior_means, prior_std_devs)
        kl_loss = distribution.kl_divergence(distrax.MultivariateNormalDiag(posterior_means, posterior_std_devs))
        kl_loss /= posterior_std_devs.shape[0]
        jax.debug.print("observation loss: {obs}, reward_loss: {rew}, kl loss: {kl}",
                        obs=observation_loss, rew=reward_loss, kl=kl_loss)
        alpha, beta, gamma = Args().args.loss_weights
        return (alpha * observation_loss + beta * reward_loss + beta * dones_loss + gamma * kl_loss,
                {
                    "info": {"observation_loss": observation_loss, "reward_loss": reward_loss, "kl_loss": kl_loss,
                             "dones_loss": dones_loss},
                    "data": (belief, state)
                })
