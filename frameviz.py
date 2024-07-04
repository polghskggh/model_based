import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    for i in range(1):
        data = np.load(f'target_{i}.npy')
        print(data.shape)
        plt.imshow(data, cmap='gray')
        plt.savefig(f'debug_{i}.png')
