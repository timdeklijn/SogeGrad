import numpy as np


class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.name = None

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_shape, output_shape, name=None):
        super().__init__()
        self.params = {
            "w": np.random.rand(input_shape, output_shape),
            "b": np.random.rand(output_shape),
        }
        self.name = name
        self.x_forward = None

    def forward(self, x):
        self.x_forward = x
        return np.dot(x, self.params["w"]) + self.params["b"]

    def backward(self, x):
        self.grads["b"] = np.sum(x, axis=0)
        self.grads["w"] = np.dot(self.x_forward.T, x)
        return np.dot(x, self.params["w"].T)


class Sigmoid(Layer):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        # TODO: should I save something here?
        return self(x)

    def backward(self, x):
        return self(x) * (1 - self(x))


class Flatten(Layer):
    def __init__(self, shape, name):
        super().__init__()
        self.shape = shape
        self.name = name

    def forward(self, x):
        return x.reshape(x.shape[0], x.shape[1] * x.shape[1])

    def backward(self, x):
        # return x.reshape(self.params["shape"])
        return x
