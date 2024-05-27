import distrax
import optax
from jax import value_and_grad

from src.models.lossfuns import mean_squared_error
from src.singletons.hyperparameters import Args
from src.trainer.trainer import Trainer
import jax.numpy as jnp


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

        data = (observations, actions, rewards, init_state, init_belief)

        loss, grads = value_and_grad(self.loss_fun, 1)(self.models, params, data)
        for key in self.models.keys():
            params[key] = optax.apply_updates(params[key], grads[key])

        print("Dreamer world loss: ", loss)
        return params

    @staticmethod
    def loss_fun(models: dict, params: dict, data: tuple):
        observations, actions, rewards, state, belief = data

        states = [] * observations.shape[0]

        key = "representation"
        for idx in range(len(states)):
            states[idx] = models[key].apply(params[key], state, actions[idx], belief, observations[idx])
            next_belief, _, _, _, posterior_states, _, _ = states[idx]

        beliefs, _, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = zip(*states)
        beliefs = beliefs.reshape(-1)
        prior_means = prior_means.reshape(-1)
        prior_std_devs = prior_std_devs.reshape(-1)
        posterior_states = posterior_states.reshape(-1)
        posterior_means = posterior_means.reshape(-1)
        posterior_std_devs = posterior_std_devs.reshape(-1)

        key = "observation"
        observation_loss = mean_squared_error(models[key], params[key], observations, beliefs, posterior_states)

        key = "reward"
        reward_loss = mean_squared_error(models[key], params[key], rewards, beliefs, posterior_states)

        distribution = distrax.MultivariateNormalDiag(prior_means, prior_std_devs)
        kl_loss = distribution.kl_divergence(distrax.MultivariateNormalDiag(posterior_means, posterior_std_devs))

        alpha, beta, gamma = Args().args.loss_weights
        return alpha * observation_loss, beta * reward_loss, gamma * kl_loss

