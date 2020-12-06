import numpy as np
from ML import plot_decision_regions

# This is a N-D nearest neighbor which uses a gradient batch decent to calculate the k(th) nearest neighbor that
# approximates the data set. Use obj = KNN(k=int) to create an instance of the class.
# The variable k is used to determine how many neighbors a value will be calculated between the each point of the
# dataset.
# X is a (n by m) numpy array and y is a 1-d by m array that holds labels for the data set.
#
# Note: This is an extremely memory hungry program. Use small datasets. I'm not your dad, so you can do the math on how
# much memory it will take to calculate the whatever size data set you are using as well as the size of each type
#
# Use obj.fit(X, y) to create copy the original data set
# Use obj.accuracy(X) to find the accuracy of the distances will print a percentage value (only works for X[0])
# Use obj.predict(X) to on a data set to see the distance between the original data set and the new data set
# Use obj.plot(X, y) to see a plot of the data
#
# Or use the included plotting function: plot_decision_regions(X, y, classifier, resolution=0.02), with the test data
# and classifier is the object of the KNN Class
class KNN(object):

    def __init__(self, k):
        # Used to hold the number of nearest neighbours
       self.k = k

    def fit(self, X, y):
        # Holds the training data
        self.x_train = X
        # Holds the labels of the data
        self.y_train = y

    def predict(self, X):
        # Holds predicted labels
        y_predicted = []

        # number of labels by flattening the training label
        numLabels = np.amax(self.y_train) + 1

        # loop through and gather all distances
        for x_test in X:
            # Find the norm of the vectors
            distances = np.sum(np.abs(self.x_train - x_test), axis=1) #np.linalg.norm(self.x_train - x_test)

            # Array to hold a list of the predicted labels
            predicted = np.zeros(numLabels, dtype=int)

            # find the correct labels using a list of indices of the shortest distances
            for i in np.argsort(distances)[:self.k]:
                # collect all the labels using the indices
                label = self.y_train[i]

                # update label array
                predicted[label] += 1

            # update the y_predicted array
            y_predicted.append(np.argmax(predicted))

        # Returns a numpy array of the predicted array
        return np.asarray(y_predicted)

    def accuracy(self, y_pred, y):
        # Prints the accuracy of the value
        print(f"Accuracy: {sum(y_pred == y) / y.shape[0]}")

    def plot(self, X, y):
        plot_decision_regions(X, y, self)