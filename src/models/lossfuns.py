import jax.numpy as jnp


def mean_squared_error(model, params, teacher_outputs, *inputs):
    return jnp.mean((model.apply(params, *inputs) - teacher_outputs) ** 2)


def compound_grad_asc(model_fixed, params_fixed, model_deriv, params_deriv, state):
    actions = model_deriv.apply(params_deriv, state)
    q_values = model_fixed.apply(params_fixed, state, actions)
    return -jnp.mean(q_values)


def cross_entropy_loss(model, params, teacher_outputs, *inputs):
    return -jnp.mean(teacher_outputs * jnp.log(model.apply(params, *inputs)))
