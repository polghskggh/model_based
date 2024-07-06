import numpy as np
from matplotlib import pyplot as plt

def show(name):
    data = np.load(name + ".npy")
    plt.imshow(data.squeeze(), cmap='gray')
    plt.savefig(name + ".png")


if __name__ == '__main__':
    for i in range(50):
        show(f'debug_simple_{i}')

