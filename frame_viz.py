import numpy as np
import jax.numpy as jnp

from matplotlib import pyplot as plt

img_array = np.load('last_predict.npy')

plt.imshow(img_array[1], cmap='gray')
plt.savefig("predict.png")

img_array = np.load('f1.npy')
idx = 1
plt.imshow(img_array[idx], cmap='gray')
plt.savefig("im1.png")


img_array = np.load('f2.npy')

plt.imshow(img_array[idx], cmap='gray')
plt.savefig("im2.png")

img_array = np.load('f3.npy')

plt.imshow(img_array[idx], cmap='gray')
plt.savefig("im3.png")

img_array = np.load('f4.npy')

plt.imshow(img_array[idx], cmap='gray')
plt.savefig("im4.png")
