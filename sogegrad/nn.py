import numpy as np
from tqdm import trange
import warnings

from sogegrad.loss import MSE
from sogegrad.optimizer import SGD

# Ignore exp overflows:
warnings.filterwarnings("ignore")


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
        optim = SGD(0.001)

        for epoch in trange(1, epochs):
            epoch_loss = 0.0
            for i in starts:
                batch = x_train[i : i + batch_size]
                predictions = self(batch)
                targets = y_train[i : i + batch_size]
                epoch_loss += loss_fn(predictions, targets)
                grad = loss_fn.gradient(predictions, targets)
                self.backward(grad)
                optim.step(self)
