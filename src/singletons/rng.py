from typing import Optional

import jax.random as jr
from singleton_decorator import singleton

from src.singletons.hyperparameters import Args


@singleton
class Key:
    initial_key: Optional[jr.PRNGKey] = None

    def key(self, number_of_keys: int) -> jr.PRNGKey:
        if self.initial_key is None:
            self.initial_key = jr.PRNGKey(Args().args.seed)

        keys = jr.split(self.initial_key, number_of_keys + 1)
        self.initial_key = keys[0]
        idx = 1 if number_of_keys == 1 else slice(1, None)
        return keys[idx]
