import gc

import distrax
import jax
import jax.numpy as jnp
from jax import value_and_grad, lax, jit

from src.enviroment import Shape
from src.models.lossfuns import reward_loss_fn, image_loss_fn, softmax_loss
from src.models.modelwrapper import ModelWrapper
from src.singletons.hyperparameters import Args
from src.singletons.rng import Key
from src.singletons.writer import log
from src.trainer.trainer import Trainer
from src.utils.rl import zero_on_term


class DreamerTrainer(Trainer):
    def __init__(self, models: dict):
        self.belief_size = Args().args.belief_size
        self.state_size = Args().args.state_size
        self.models = models

    def apply_grads(self, grads):
        for key in grads.keys():
            self.models[key].apply_grads(grads[key])

    def train_step(self, initial_belief, initial_state, observations, actions, rewards, dones):
        rng = ModelWrapper.make_rng_keys()
        keys_to_select = ['representation', 'observation', 'reward', 'encoder']

        if Args().args.predict_dones:
            keys_to_select.append('dones')

        params = {key: self.models[key].params for key in keys_to_select}
        models = {key: self.models[key].model for key in keys_to_select}

        batch_size = Args().args.batch_size
        for _ in range(Args().args.num_epochs):
            for env_idx in range(0, observations.shape[0]):
                last_belief, last_state = initial_belief[env_idx], initial_state[env_idx]
                for start_idx in range(0, observations.shape[1], batch_size):
                    batch_slice = slice(start_idx, start_idx + batch_size)
                    data = (observations[env_idx][batch_slice], actions[env_idx][batch_slice],
                            rewards[env_idx][batch_slice], dones[env_idx][batch_slice],
                            last_state[env_idx], last_belief[env_idx])
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

        beliefs = jnp.zeros((observations.shape[0], Args().args.belief_size))
        state_shape = (observations.shape[0], Args().args.state_size)
        prior_means = jnp.zeros(state_shape)
        prior_std_devs = jnp.zeros(state_shape)
        states = jnp.zeros(state_shape)
        posterior_means = jnp.zeros(state_shape)
        posterior_std_devs = jnp.zeros(state_shape)

        key = "encoder"
        encoded_observations = jit(models[key].apply)(params[key], observations, rngs=rng)

        key = "representation"
        for idx in range(len(states)):
            output = jit(models[key].apply)(params[key], jnp.expand_dims(state, 0),
                                            jnp.expand_dims(actions[idx], 0),
                                            jnp.expand_dims(belief, 0),
                                            jnp.expand_dims(encoded_observations[idx], 0), rngs=rng)[0]
            belief = output[0]
            state = output[1]

            beliefs = beliefs.at[idx].set(output[0])
            states = states.at[idx].set(output[1])

            prior_means = prior_means.at[idx].set(output[2])
            prior_std_devs = prior_std_devs.at[idx].set(output[3])
            posterior_means = posterior_means.at[idx].set(output[4])
            posterior_std_devs = posterior_std_devs.at[idx].set(output[5])

        prior_means = prior_means.reshape(-1)
        prior_std_devs = prior_std_devs.reshape(-1)
        posterior_means = posterior_means.reshape(-1)
        posterior_std_devs = posterior_std_devs.reshape(-1)

        key = "observation"
        pixels = jit(models[key].apply)(params[key], beliefs, states, rngs=rng)
        observation_loss = image_loss_fn(pixels, observations)

        key = "reward"
        reward_logits = jit(models[key].apply)(params[key], beliefs, states)
        reward_loss = reward_loss_fn(reward_logits, rewards)

        dones_loss = 0
        if Args().args.predict_dones:
            key = "dones"
            dones_logits = jit(models[key].apply)(params[key], beliefs, states)
            dones_loss = jnp.mean(softmax_loss(dones_logits, dones))

        distribution = distrax.MultivariateNormalDiag(prior_means, prior_std_devs)
        kl_loss = distribution.kl_divergence(distrax.MultivariateNormalDiag(posterior_means, posterior_std_devs))
        kl_loss /= posterior_std_devs.shape[0]

        alpha, beta, gamma = Args().args.loss_weights
        return (alpha * observation_loss + beta * reward_loss + beta * dones_loss + gamma * kl_loss,
                {
                    "info": {"observation_loss": observation_loss, "reward_loss": reward_loss, "kl_loss": kl_loss,
                             "dones_loss": dones_loss},
                    "data": (belief, state)
                })
