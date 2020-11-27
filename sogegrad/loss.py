class Loss:
    def forward(self, pred, actual):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError


class MSE:
    def forward(self, pred, actual):
        assert pred.shape == actual.shape
        return np.sum((pred - actual) ** 2, axis=-1)

    def backward(self, pred, actual):
        return pred - actual