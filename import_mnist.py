#Based on https://mattpetersen.github.io/load-mnist-with-numpy
import gzip
import os
import numpy as np

files = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']

def load(path=None):
    if path is None:
       path = os.path.join(os.getcwd(), 'mnist')

    def images(path):
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        return pixels.reshape(-1, 28,28)

    def labels(path):
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            integer_labels = np.frombuffer(f.read(), 'B', offset=8)
        return integer_labels

    train_images = images(os.path.join(path, files[0]))
    train_labels = labels(os.path.join(path, files[1]))
    test_images = images(os.path.join(path, files[2]))
    test_labels = labels(os.path.join(path, files[3]))

    return train_images, train_labels, test_images, test_labels