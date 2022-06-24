import matplotlib.pylab as plt
import numpy as np
import torch


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


def load(train: bool = True):

    global IMAGES, LABELS

    img_file = "./test_data/t10k-images.idx3-ubyte"
    label_file = "./test_data/t10k-labels.idx1-ubyte"

    if train:
        img_file = "./train_data/train-images.idx3-ubyte"
        label_file = "./train_data/train-labels.idx1-ubyte"

    with open(img_file, "rb") as f:
        print("---Reading Images---")
        data = f.read()
        num_of_images = int.from_bytes(data[4:8], "big")
        IMAGES = np.array_split(np.array([i for i in data[16:]]), num_of_images)

    with open(label_file, "rb") as f:
        print("---Reading Labels---")
        LABELS = f.read()[8:]

    return num_of_images
