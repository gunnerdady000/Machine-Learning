import numpy as np
from ML import plot_decision_regions

# SoftSVM class that uses linear algebra to calculate a line of best fit that attempts to split the data. To
# initialize please use: SoftSVM(rate = float, niter=float, lamda=float) and assign to an object.
# X is an array of real values in the shape of X:X[0]|X[1] and y is an array the same length of X[0] and must be
# filled with either 1 or -1. Use ndarrays for all the two vectors.
#
# Use obj.fit(X, y) where X is a (2 by n) dimensional numpy array and y is a (1 by n) dimensional numpy array using test
# test data to find a linear line between the data as well as the bounds of the dataset.
# Use obj.predict(X) to use the fitted line on a data set, where X is a (2 by n) dimensional numpy array. Returns a
# (1 by n) dimensional numpy array.
# Use obj.plot(X, y) to show a border wall between the data sets
#
class SoftSVM(object):
    def __init__(self, rate=0.001, niter=1000, lamda=0.0001):
        # learning rate
        self.rate = rate
        # number of iterations
        self.niter = niter
        # Lambda * 2, as 1 - -1 = 2, smaller value = Hard SVM
        self.lamda = 2 * lamda

    def fit(self, X, y):
        # Fit training data X : Training vectors, X.shape : [#samples, #features] y : Target values, y.shape : [#samples]
        x_samples, features = X.shape

        # weights: create a weights array of right size and intialize
        self.weight = np.zeros(features)

        # create the intercept location
        self.b = 0

        # main loop to fit the data to the labels
        for i in range(self.niter):
            for index, xi in enumerate(X):

                # Implemented Dual -> Quadratic from https://en.wikipedia.org/wiki/Support_vector_machine#Hard-margin
                # This is looking for the smallest nonzero value that separates the delta of the data
                if y[index] * (np.dot(xi, self.weight) - self.b) >= 1:
                    # waking around the derivative of the line
                    self.weight -= self.rate * (self.lamda * self.weight)
                    # setting b = 0 as the math guides
                    self.b = 0
                else:
                    # as the math says we need to continue to minimize the delta
                    self.weight -= self.rate * (self.lamda * self.weight - np.dot(xi, y[index]))
                    # as the math guides, I will subtract the b by the learning rate times the current y-value
                    self.b -= self.rate * y[index]

        return self

    def predict(self, X):
        # (w0, b0) = argmin ||w||^2 s.t. for all i yi(<w,xi>+b)>= 1
        return np.sign(np.dot(X, self.weight) + self.b)

    def plot(self, X, y):
        plot_decision_regions(X, y, self)