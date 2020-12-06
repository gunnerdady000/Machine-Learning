import numpy as np
from math import sqrt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from ML import plot_decision_regions


# Perceptron class that uses linear algebra to calculate a line of best fit that attempts to split the data. To
# initialize please use: Percpetron(rate = float, niter=float) and assign to an object. X is an array of real values in
# the shape of X:X[0]|X[1] and y is an array the same length of X[0] and must be filled with either 1 or -1. Use
# ndarrays for all the two vectors.
# Use obj.fit(X, y) where X is a (2 by n) dimensional numpy array and y is a (1 by n) dimensional numpy array using test
# test data to find a linear line between the data.
# Use obj.predict(X) to use the fitted line on a data set, where X is a (2 by n) dimensional numpy array. Returns a
# (2 by n) dimensional numpy array.
#
# Use the included plotting function: plot_decision_regions(X, y, classifier, resolution=0.02), with the test data
# and classifier is the object of the Precpectron Class
class Perceptron(object):
    def __init__(self, rate=0.01, niter=10):
        self.rate = rate  # learning rate
        self.niter = niter  # number of iterations

    def fit(self, X, y):
        # Fit training data X : Training vectors, X.shape : [#samples, #features] y : Target values, y.shape : [#samples]"""

        # determines if we need to stop more training cycles because we have learned all we can
        convergence = 0

        # weights: create a weights array of right size and intialize
        self.weight = np.zeros(1 + X.shape[1])

        # Number of misclassifications, creates an array to hold the number of misclassifications
        self.errors = []

        # main loop to fit the data to the labels
        for i in range(self.niter):
            # set iteration error to zero
            error = 0

            # loop over all the objects in X and corresponding y element
            for xi, target in zip(X, y):
                # calculate the needed (delta_w) update from previous step
                # delta_w = rate * (target – prediction current object)
                delta_w = self.rate * (target - self.predict(xi))

                # calculate what the current object will add to the weight Why not self.weight[1:] += rate * (target - predict) * xi?
                self.weight[1:] += delta_w * xi

                # set the bias to be the current delta_w why not do self.weight[0] += rate*(target - predict)?
                self.weight[0] += delta_w

                # increase the iteration error if delta_w != 0
                error += int(delta_w != 0.0)

            # Update the misclassification array with # of errors in iteration
            self.errors.append(error)

            # add in check if we have reached 0 error
            if self.errors[i] == 0:
                # print("Convergence: ", convergence) #testing only
                convergence = convergence + 1
                # print("Convergence: ", convergence) #testing only
                if convergence == 2:
                    return self
        # return self
        return self

    def net_input(self, X):
        """Calculate net input"""
        # return the return the dot product: X.w + bias
        return np.dot(X, self.weight[1:]) + self.weight[0]

    def predict(self, X):
        # Return class label after unit step
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def plot(self, X, y):
        plot_decision_regions(X, y, self)