import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    for i in range(16):
        data = np.load(f'debug_{i}.npy')
        plt.imshow(data, cmap='gray')
        plt.savefig(f'debug_{i}.png')
