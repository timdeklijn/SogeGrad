import numpy as np


class Loss:
    def __call__(self, pred, actual):
        raise NotImplementedError

    def gradient(self, x):
        raise NotImplementedError


class MSE:
    def __call__(self, pred, actual):
        assert pred.shape == actual.shape
        return np.sum((pred - actual) ** 2)

    def gradient(self, pred, actual):
        return 2 * (pred - actual)
