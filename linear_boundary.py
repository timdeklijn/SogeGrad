import numpy as np

from sogegrad.nn import Model
from sogegrad.layer import Linear


def f(x, y):
    """
    Create 'labels' from the following:

        y = mx + b

    Simply return 0 if y(x) is below the line or 1 if y(x) is above the line
    """
    # Slope and intercept:
    m = 0.4
    b = 0.2
    # Calculate true value
    yy = m * x + b
    # Assess if input y is over or under true value
    if yy >= y:
        return 0
    return 1


def create_xy(size=100):
    """
    Create a set of xs and ys with size 'size' based on f(x,y) output.
    """
    # create a list of x's and y's of length size.
    xs = np.dstack((np.random.rand(size), np.random.rand(size)))[0]
    # Create data labels.
    ys = np.array([np.array([f(x[0], x[1])]) for x in xs])
    return xs, ys


# Create train and test data
x_train, y_train = create_xy(1000)
x_test, y_test = create_xy(50)

# Create model
model = Model([Linear(2, 1, "linear1")])

# Train model.
model.fit(x_train, y_train, epochs=1000, batch_size=10)

for i in range(10):
    sample = x_test[i]
    print(sample, np.abs(np.rint(model(sample))), y_test[i])
