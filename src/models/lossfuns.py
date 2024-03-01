import jax.numpy as jnp


def mean_squared_error(model, params, inputs, teacher_outputs):
    return jnp.mean((model.apply(params, inputs) - teacher_outputs) ** 2)


loss_funs = {
    "mse": mean_squared_error
}
