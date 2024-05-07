import distrax
import optax
from jax import value_and_grad

from src.models.lossfuns import mean_squared_error
from src.pod.hyperparameters import hyperparameters
from src.trainer.trainer import Trainer
import jax.numpy as jnp


class DreamerTrainer(Trainer):
    def __init__(self, representation_model, observation_model, reward_model):
        self.batch_size = hyperparameters["dreamer"]["batch_size"]
        self.belief_size = hyperparameters["dreamer"]["belief_size"]
        self.state_size = hyperparameters["dreamer"]["state_size"]

        self.models = {"representation": representation_model,
                       "observation": observation_model,
                       "reward": reward_model}

    def train_step(self, observations, actions, rewards, nonterminals, **params):
        init_belief = jnp.zeros(self.batch_size, self.belief_size)
        init_state = jnp.zeros(self.batch_size, self.state_size)

        data = (observations, actions, rewards, nonterminals, init_state, init_belief)

        loss, grads = value_and_grad(self.loss_fun, 1)(self.models, params, data)
        for key in self.models.keys():
            params[key] = optax.apply_updates(params[key], grads[key])

        print("Dreamer world loss: ", loss)
        return params

    @staticmethod
    def loss_fun(data: tuple, models: dict, params: dict):
        observations, actions, rewards, nonterminals, init_state, init_belief = data

        key = "representation"
        state = models[key].apply(params[key], init_state, actions, init_belief, observations, nonterminals)
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = state

        key = "observation"
        observation_loss = mean_squared_error(models[key], params[key], observations, beliefs, posterior_states)

        key = "reward"
        reward_loss = mean_squared_error(models[key], params[key], rewards, beliefs, posterior_states)

        distribution = distrax.MultivariateNormalDiag(prior_means, prior_std_devs)
        kl_loss = distribution.kl_divergence(distrax.MultivariateNormalDiag(posterior_means, posterior_std_devs))

        alpha, beta, gamma = hyperparameters["dreamer"]["loss_weights"]
        return alpha * observation_loss, beta * reward_loss, gamma * kl_loss
