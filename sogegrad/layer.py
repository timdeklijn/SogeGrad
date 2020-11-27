import numpy as np


class Layer:
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_shape, output_shape):
        self.params = {
            "w": np.zeros((input_shape, output_shape)),
            "b": np.zeros((output_shape)),
        }
        self.grads = {}

    def forward(self, x):
        return np.dot(x, self.params["w"]) + self.params["b"]

    def backward(self, x):
        self.grads["b"] = np.sum(x, axis=0)
        self.grads["w"] = np.dot(
            self.inputs.T,
        )
        return np.dot(x, self.params["w"].T)


class Sigmoid(Layer):
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        return self.sigmoid(x)

    def backward(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
