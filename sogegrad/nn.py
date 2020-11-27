class Model:
    def __init__(self, sequence):

        self.model = sequence

    def __call__(self, x):
        for layer in self.model:
            x = layer.forward(x)
        return x

    def backward(self, x):
        for layer in reversed(self.model):
            x = layer.backward(x)
        return x

    def fit(self, x_train, y_train):
        pass