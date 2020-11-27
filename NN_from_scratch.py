# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# %%
# Import helper functions
from helpers import plot_number

# %%
def load_mnist_data():
    """Download mnist dataset, scale the images"""
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test


# %%
# Load train + test data
x_train, y_train, x_test, y_test = load_mnist_data()
# set size, used later
size = x_train[0].shape

# %%
plot_number(x_train[0])

# %%

# Define a sequential model that will flatten an image, run the data through a hidden
# layer with a sigmoid ctivation function to an output layer, also with a sigmoid
# activation function.
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="sigmoid"),
        tf.keras.layers.Dense(10, activation="sigmoid"),
    ]
)

# %%
# Define a loss function and compile the model.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

# %%
# Train the model
model.fit(x_train, y_train, epochs=10)

# %%
# Evaluate the model
model.evaluate(x_test, y_test, verbose=2)

# %%
# Run a random sample from the test set through the model and check the prediction.
n = np.random.randint(0, len(x_test))
plot_number(x_test[n])
prediction = np.argmax(model(x_test[n].reshape(1, size[0], size[1])).numpy())
print("Prediction:", prediction, "- Actual:", y_test[n])

# %%
def extract_weights_from_model(model):
    """Extract the weights from a model and add the to a list"""
    weights = []
    for l in model.get_weights():
        weights.append(l)
        print(l.shape)
    return weights


# Extract the weights from the model
weights = extract_weights_from_model(model)

# %%

# Our custom NN class. This only works for a simple feed forward NN.
class NN:
    @staticmethod
    def sigmoid(i):
        return 1 / (1 + np.exp(-i))

    @staticmethod
    def relu(i):
        return np.maximum(0, i)

    @staticmethod
    def dense(x, kernel, bias, activation):
        return activation(np.dot(x, kernel) + bias)


class MyNN(NN):
    def __init__(self, weights):
        self.weights = weights

    def predict(self, x):
        x = x.flatten()
        x = self.dense(x, self.weights[0], self.weights[1], self.sigmoid)
        x = self.dense(x, self.weights[2], self.weights[3], self.sigmoid)
        return np.argmax(x)


# %%

# Check a random sample from the test set with our own model.
n = np.random.randint(0, len(x_test))
plot_number(x_test[n])
model = MyNN(weights)
prediction = model.predict(x_test[n])
print("Prediction:", prediction, "- Actual:", y_test[n])

# %%
