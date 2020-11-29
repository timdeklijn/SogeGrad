# def next_batch(X, y, batchSize):
# 	# loop over our dataset `X` in mini-batches of size `batchSize`
# 	for i in np.arange(0, X.shape[0], batch_size):
# 		# yield a tuple of the current batched data and labels
# 		yield (X[i:i + batch_size], y[i:i + batch_size])

# def weight_update(batch_x, error, learning_rate, weights):
#     # gradient update is the dot product between the transpose of
#     #our current batch and the error on the batch
#     gradient = batch_x.T.dot(error) / batch_x.shape[0]

#     #use the computed gradient on the current batch to take
#     #a "step" in the correct direction
#     weights += -learning_rate * gradient

#     return weights


# lossHistory = []
# # loop over the desired number of epochs
# for epoch in np.arange(0, epochs):
# 	# initialize the total loss for the epoch
# 	epoch_loss = []
# 	# loop over our data in batches
# 	for (batchX, batchY) in next_batch(X, y, batch_size):
# 		# take the dot product between our current batch of
# 		# features and weight matrix `W`, then pass this value
# 		# through the sigmoid activation function
# 		preds = sigmoid_activation(batchX.dot(W))
# 		# now that we have our predictions, we need to determine
# 		# our `error`, which is the difference between our predictions
# 		# and the true values
# 		error = preds - batchY
# 		# given our `error`, we can compute the total loss value on
# 		# the batch as the sum of squared loss
# 		loss = np.sum(error ** 2)
# 		epoch_loss.append(loss)

#         #compute new weights using gradient descent
#         W = weight_update(batchX, error, learning_rate, W)

# 	# update our loss history list by taking the average loss
# 	# across all batches
# 	lossHistory.append(np.average(epoch_loss))


class Optim:
    def step(self, model):
        raise NotImplementedError


class SGD(Optim):
    def __init__(self, lr):
        self.lr = lr

    def step(self, model):
        for layer in model.model:
            if layer.params:
                for k in layer.params.keys():
                    layer.params[k] -= self.lr * layer.grads[k]
