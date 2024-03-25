import pip
from jax import devices


def setup_gpu():
    pip.main(['install', '--upgrade', 'jax[cuda12_local]', '-f',
              'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'])
    check_gpu()


def check_gpu():
    try:
        gpu_devices = devices('gpu')
        print("GPU is available.")
        for gpu in gpu_devices:
            print(gpu)
    except RuntimeError:
        print("No GPU available.")

