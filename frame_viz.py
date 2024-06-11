import numpy as np
import jax.numpy as jnp

img_array = np.load('last_predict.npy')
next_frames = jnp.argmax(img_array, axis=-1, keepdims=True)

from matplotlib import pyplot as plt

plt.imshow(next_frames[0], cmap='gray')
plt.savefig("averaged_image.png")
plt.show()
