import jax.numpy as jnp


def rebatch(batch_size, *data):
    """
    input_dims: (batch1, batch2, data_dims)
    output_dims: list of (batch, data_dims)

    :param batch_size: The size of the second (batch) dimension of the data
    :param data: The data to be rebatched
    :return: The rebatched data
    """
    def _rebatch(data_array):
        single_batch_dim = data_array.reshape(-1, *data_array.shape[2:])
        return jnp.array_split(single_batch_dim, single_batch_dim.shape[0] // batch_size)

    return (_rebatch(data_array) for data_array in data)
