import numpy as np

from sogegrad.nn import Model
from sogegrad.layer import Sigmoid, Linear, Flatten


def f(x, y):
    # y = mx + b
    # m = 2
    # b = 2
    #
    # Simply return 0 if y(x) is below the line or 1 if y(x) is above the line
    yy = 1 * x + 0
    if yy >= y:
        return 0
    return 1


def create_xy(size=100):
    # create a list of x's and y's of length size.
    xs = np.dstack((np.random.rand(size), np.random.rand(size)))[0]
    ys = np.array([np.array([f(x[0], x[1])]) for x in xs])
    return xs, ys


x_train, y_train = create_xy(10000)
x_test, y_test = create_xy(100)


model = Model([Linear(2, 1, "linear1")])

model.fit(x_train, y_train, 1000, 100)