import jax.numpy as jnp
from jax import jit


def mean_squared_error(model, params, teacher_outputs, *inputs):
    return jnp.mean((jit(model.apply)(params, *inputs) - teacher_outputs) ** 2)


def compound_grad_asc(model_fixed, params_fixed, model_deriv, params_deriv, state):
    actions = jit(model_deriv.apply)(params_deriv, state)
    q_values = jit(model_fixed.apply)(params_fixed, state, actions)
    return jnp.mean(q_values)


loss_funs = {
    "mse": mean_squared_error,
    "compound_grad_asc": compound_grad_asc
}
