import tensorflow as tf
import numpy as np

from sogegrad.nn import Model
from sogegrad.layer import Sigmoid, Linear, Flatten


def download_mnist():
    """
    Download and scale mnist dataset
    """
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test


def dummy_y(y):
    """
    From a list of [i]'s, create an array of [....] where
    all numbers are 0 except the i'th number.
    """
    ret = np.zeros((len(y), 10))
    for i in range(len(y)):
        ret[i][y[i]] = 1
    return ret


x_train, y_train, x_test, y_test = download_mnist()
# Modify y to have dummies
y_train = dummy_y(y_train)
y_test = dummy_y(y_test)

# Select a small subset of the whole dataset
subset = True
if subset:
    x_train = x_train[:1000]
    y_train = y_train[:1000]

# Batch size of training
batch_size = 5

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

# Train the model
model.fit(x_train, y_train, 1000, batch_size)

# Predict some of the test data and print the results
for i in range(10):
    prediction = model(np.array([x_test[i]]))
    print(np.argmax(prediction), np.argmax(y_test[i]))
