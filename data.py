import threading
from time import sleep

import matplotlib.pylab as plt
import numpy as np
import torch

INDEX = 0


def change_scale(
    old_value, old_min: int = 0, old_max: int = 255, new_min: int = 0, new_max: int = 1
) -> int:

    return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


def show_next():

    for i, img in enumerate(IMAGES):
        yield np.reshape(img, (28, 28)), str(LABELS[i])


def get_next():

    for i, img in enumerate(IMAGES):
        yield torch.reshape(torch.tensor(img), (1, 784)).float(), torch.reshape(
            torch.tensor([0 if LABELS[i] != indx else 1 for indx in range(10)]), (1, 10)
        ).float()


with open("train-images.idx3-ubyte", "rb") as f:
    print("---Reading Images---")
    data = f.read()
    IMAGES = np.array_split(
        np.array([change_scale(i) for i in data[16:]]), int.from_bytes(data[4:8], "big")
    )


with open("train-labels.idx1-ubyte", "rb") as f:
    print("---Reading Labels---")
    LABELS = f.read()[8:]


FIG = plt.gcf()


# #### Miscellaneous functions
# def sigmoid(z):
#     """The sigmoid function."""
#     return 1.0 / (1.0 + np.exp(-z))


# def sigmoid_prime(z):
#     """Derivative of the sigmoid function."""
#     return sigmoid(z) * (1 - sigmoid(z))
