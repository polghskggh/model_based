import numpy as np
import jax.numpy as jnp

from matplotlib import pyplot as plt


idx = 7
img_array = np.load('f1.npy')
print(jnp.mean(img_array))
plt.imshow(img_array[idx], cmap='gray', vmin=0, vmax=255)
plt.savefig("im1.png")


img_array = np.load('f2.npy')

plt.imshow(img_array[idx], cmap='gray', vmin=0, vmax=255)
plt.savefig("im2.png")

img_array = np.load('f3.npy')

plt.imshow(img_array[idx], cmap='gray', vmin=0, vmax=255)
plt.savefig("im3.png")

img_array = np.load('f4.npy')

plt.imshow(img_array[idx], cmap='gray', vmin=0, vmax=255)
plt.savefig("im4.png")
img_array = np.load('f5.npy')

plt.imshow(img_array[idx], cmap='gray', vmin=0, vmax=255)
plt.savefig("im5.png")
