import numpy as np
import jax.numpy as jnp

from matplotlib import pyplot as plt

img_array = np.load('last_predict.npy')

plt.imshow(img_array[1], cmap='gray')
plt.savefig("predict.png")

img_array = np.load('teach_frame.npy')


plt.imshow(img_array[10], cmap='gray')
plt.savefig("teach.png")

img_array = np.load('frame1.npy')


plt.imshow(img_array, cmap='gray')
plt.savefig("f1.png")

img_array = np.load('frame2.npy')


plt.imshow(img_array, cmap='gray')
plt.savefig("f2.png")

img_array = np.load('frame4.npy')


plt.imshow(img_array, cmap='gray')
plt.savefig("f4.png")

img_array = np.load('frame_last_reshaped.npy')


plt.imshow(img_array, cmap='gray')
plt.savefig("fr.png")
