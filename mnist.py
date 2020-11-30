import tensorflow as tf

from sogegrad.nn import Model
from sogegrad.layer import Sigmoid, Linear, Flatten


def download_mnist():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = download_mnist()


x_train = x_train[:1000]
y_train = y_train[:1000]

batch_size = 100
# Define a model
model = Model(
    [
        Flatten((28, 28), "flatten"),
        Linear(28 * 28, 128, "linear1"),
        Sigmoid("sigmoid1"),
        Linear(128, 10, "linear2"),
        Sigmoid("sigmoid2"),
    ]
)
model.fit(x_train, y_train, 100, batch_size)
# prediction = model(x_test[0])
