import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    for i in range(10):
        data = np.load(f'target_{i}.npy')
        print(data.shape)
        plt.imshow(data.squeeze(), cmap='gray')
        plt.savefig(f'target_{i}.png')
        # data = np.load(f'debug_{i}.npy')
        # plt.imshow(data.squeeze(), cmap='gray')
        # plt.savefig(f'debug_{i}.png')
