from jax import tree_map


def sum_dicts(dict1, dict2):
    return tree_map(lambda x, y: x + y, dict1, dict2)
