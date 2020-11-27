import tensorflow as tf

from sogegrad.nn import Model
from sogegrad.layer import Sigmoid, Linear


def download_mnist():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test


# Define a model
model = Model([Linear(28 * 28, 128), Sigmoid(), Linear(128, 10), Sigmoid()])

x_train, y_train, x_test, y_test = download_mnist()

model.fit(x_train, y_train)

prediction = model(x_test[0])

# def train(x_train, y_train):


# import matplotlib.pyplot as plt
