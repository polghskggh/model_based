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
        apply_funs = {key: jit(self.models[key].model.apply) for key in keys_to_select}

        batch_size = Args().args.batch_size
        grad_fn = value_and_grad(self.loss_fun, 1, True)
        for _ in range(Args().args.num_epochs):
            for env_idx in range(0, observations.shape[1]):
                last_belief, last_state = initial_belief[env_idx], initial_state[env_idx]
                for start_idx in range(0, observations.shape[0], batch_size):
                    batch_slice = slice(start_idx, start_idx + batch_size)
                    (loss, aux), grads = grad_fn(apply_funs, params,
                                                 jnp.expand_dims(observations[batch_slice, env_idx], 1),
                                                 jnp.expand_dims(actions[batch_slice, env_idx], 1),
                                                 jnp.expand_dims(rewards[batch_slice, env_idx], 1),
                                                 jnp.expand_dims(dones[batch_slice, env_idx], 1),
                                                 jnp.expand_dims(last_state, 0),
                                                 jnp.expand_dims(last_belief, 0), rng=rng)
                    self.apply_grads(grads)
                    last_belief, last_state = aux["data"]
                    log(aux["info"])
                    print("backwards pass completed")

        new_params = {"params": self.models["representation"].params["params"]["transition_model"]}
        self.models["transition"].params = new_params
        return self.models

    @staticmethod
    def loss_fun(apply_funs: dict, params: dict, observations, actions, rewards, dones, state, belief, rng: dict):
        key = "encoder"
        encoded_observations = apply_funs[key](params[key], observations, rngs=rng)


        key = "representation"
        def scan_fn(carry, inputs):
            action, encoded_observation = inputs
            belief_carry, state_carry = carry

            step_output = apply_funs[key](params[key], state_carry,
                                          action, belief_carry, encoded_observation, rngs=rng)

            return (step_output[0], step_output[1]), step_output
            
        _, output = jax.lax.scan(scan_fn, (belief, state), (actions, encoded_observations)) 
        beliefs, states, prior_means, prior_std_devs, posterior_means, posterior_std_devs = (output[0], output[1],
                                                                                             output[2], output[3],
                                                                                             output[4], output[5])
        beliefs = beliefs.reshape(-1, beliefs.shape[-1])
        states = states.reshape(-1, states.shape[-1])
        prior_means = prior_means.reshape(-1)
        prior_std_devs = prior_std_devs.reshape(-1)
        posterior_means = posterior_means.reshape(-1)
        posterior_std_devs = posterior_std_devs.reshape(-1)

        key = "observation"
        pixels = apply_funs[key](params[key], beliefs, states, rngs=rng)
        observation_loss = image_loss_fn(pixels, observations)

        key = "reward"
        reward_logits = apply_funs[key](params[key], beliefs, states)
        reward_loss = reward_loss_fn(reward_logits, rewards)

        dones_loss = 0
        if Args().args.predict_dones:
            key = "dones"
            dones_logits = apply_funs[key](params[key], beliefs, states)
            dones_loss = jnp.mean(softmax_loss(dones_logits, dones))

        distribution = distrax.MultivariateNormalDiag(prior_means, prior_std_devs)
        kl_loss = distribution.kl_divergence(distrax.MultivariateNormalDiag(posterior_means, posterior_std_devs))
        kl_loss /= posterior_std_devs.shape[0]

        alpha, beta, gamma = Args().args.loss_weights
        return (alpha * observation_loss + beta * reward_loss + beta * dones_loss + gamma * kl_loss,
                {
                    "info": {"observation_loss": observation_loss, "reward_loss": reward_loss, "kl_loss": kl_loss,
                             "dones_loss": dones_loss},
                    "data": (beliefs[-1], states[-1])
                })
