import numpy as np

from sogegrad.loss import MSE
from sogegrad.optimizer import SGD


class Model:
    def __init__(self, layers):
        self.model = layers

    def __call__(self, x):
        for layer in self.model:
            x = layer.forward(x)
        return x

    def backward(self, x):
        for layer in reversed(self.model):
            x = layer.backward(x)
        return x

    def fit(self, x_train, y_train, epochs, batch_size=10):
        # TODO: shuffle data?
        starts = np.arange(0, len(x_train), batch_size)
        loss_fn = MSE()
        optim = SGD(0.01)

        for epoch in range(1, epochs):
            epoch_loss = 0.0
            for i in starts:
                batch = x_train[i : i + batch_size]
                predictions = self(x_train[i : i + batch_size])
                targets = dummy_y(y_train[i : i + batch_size])
                epoch_loss += loss_fn(predictions, targets)
                grad = loss_fn.gradient(predictions, targets)
                self.backward(grad)
                optim.step(self)
            print(epoch, epoch_loss)


def dummy_y(y):
    ret = np.zeros((len(y), 10))
    for i in range(len(y)):
        ret[i][y[i]] = 1
    return ret