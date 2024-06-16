from sys import argv

import numpy as np
import jax.numpy as jnp

from matplotlib import pyplot as plt


idx = 7

for i in range(int(argv[1])):
    img_array = np.load(f"f{i}.npy")
    print(jnp.mean(img_array))
    plt.imshow(img_array[idx], cmap='gray', vmin=0, vmax=255)
    plt.savefig(f"im{i}.png")

