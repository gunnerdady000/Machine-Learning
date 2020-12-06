# ML.py
import numpy as np
from math import sqrt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger


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


# Single linear regression without using a matrix. To initialize the user simply has to use: LinearRegression() and
# assign to an object. All x & y are 1-dimensional numpy arrays.
# Use obj.predict(x,y) where x and y are single-dimensional data.
# Use obj.graph(x,y) to graph the predicted trend-line on a scatter plot.
# Use obj.root_error to see the Root-Mean-Squared-Error.
# Use obj.weight to see both the intercept and slope, respectively to their position in the array. y = mx + b
#
# Note: User can use a different 1-d x and y data set and compare it against the predicted trend-line, I'm not your dad.
class LinearRegression(object):
    # Initializes the class
    def __init__(self):
        # create an empty weights, size 2, weight[0] = b, and weight[1] = m
        self.weight = np.empty([2, 1])

        # Create the root mean square variable
        self.root_error = 0

    # Make user use a numpy array for X and y. Creates weights to find a line of best fit
    def predict(self, x, y):
        # m = Sum( (x - mean(x)) * (y - mean(y)) / Sum( (x - mean(x)^2) )
        # This equation can be found on here on the Wikipedia page over Simple linear regression
        # https://en.wikipedia.org/wiki/Simple_linear_regression

        #############################################################################################
        # If numpy did not exist this how to calculate, this how you can create the slope (m)
        # From the math above we can save time by calculating X - mean(x) and then multiple it by y
        # and squaring itself into separate variables this would allow one loop.
        # One can also calculate the mean = (Sum(list)/size(list)), this would be a total of 2 loops
        ##############################################################################################

        # I am going to let numpy take care of the equation
        # calculate the mean of x and Y
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        # We will now calculate the slope
        self.weight[1] = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x) ** 2)

        # We will now make the intercept, which is b = (mean(y) - m * mean(x))
        self.weight[0] = mean_y - self.weight[1] * mean_x

        # Thanks for coming to my TED Talk, we can now find the R value be able to plot this line
        # We will now store a predicted trend line, so we can standardize our classes

        # This will loop through and create a list of y values based off of our predicted regression line, tried doing
        # an inline command that I used in COA program, but it got anger at me, so a function will do
        self.predicted_regression = self.approximate_y(x)

        # call the root mean square function
        self.root_error = self.rmse(y)

    # Uses the weights from the prediction to give create a predicted y-values
    def approximate_y(self, x):
        # loop through the x, values and return an array that holds those values
        return self.weight[1] * x + self.weight[0]

    # Used to get the Root-mean-squared-error
    def rmse(self, y):
        # This equation can also be found in the same wikipedia article, but it finds the mean between the
        # predicted y outputs and the actual y outputs using the following equation:
        # rmse = sqrt ( Sum(y_predicted - y)^2 / (total amount of numbers) )
        error = np.sum((self.predicted_regression - y) ** 2) / len(y)
        return sqrt(error)

    # Used to graph the scatter plot of the original data as well as our predicted line
    def graph(self, x, y):
        # create a scatter plot with the given dataset of x and y
        # Here is a handy page from a website that tells people how to use scatter plots
        # https://www.machinelearningplus.com/plots/python-scatter-plot/

        # Just realized we need to update the error for any given data set and not just the trained data set
        self.root_error = self.rmse(y)

        # They showed how to add in a label for each line and add in test
        plt.scatter(x, y, label=f'True y correlation = {1}')
        plt.scatter(x, self.predicted_regression,
                    label=f'RMSE = {np.round(self.root_error, 4)}, Eq = {np.round(self.weight[1], 4)}x + {np.round(self.weight[0], 4)}')
        plt.plot(x, self.weight[1] * x + self.weight[0], '-', color='orange')
        # create the legend and plot it to user
        plt.legend(loc='best')
        plt.show()


# This is a 1-D threshold learner which uses a bruteforce method to find a threshold-value that matches a 1-D dataset
# as closely as possible. To initialize use ThresholdLearner(minthreshold, maxthreshold, iteration, b) and assign to an
# object. The following will explain what each of the variables do:
# minthreshold: is the user's lowest or starting threshold-value
# maxthreshold: is the user's highest or ending threshold-value
# iteration: is the user's number of times the threshold will increment through the threshold. The program will exit if
#            a perfect threshold-value is found before the number of iterations have been reached :)
# b: is the value either (1) or (-1) which will determine what type of desired binary data is being targeted
# Note: this program will auto calculate the precision of the increment using the following equation:
#       increment = (maxthreshold - minthreshold) / iteration
# If more precision is required, increase the number of iterations for a smaller threshold step :)
#
# Use obj.predict(x, y) to find the closest threshold that can represent a 1-D dataset given desired outputs are binary
# not (base2) but either 1, or -1
# x: should be a 1-d numpy array of float/double values
# y: should be 1-d numpy array of 1 or -1 (any integer value may be used in place of -1)
# obj.error: will give the list of errors calculated between each iteration
# obj.threshold: will give the user the predicted threshold-value
#
# Use graph(x, y) to get a scatter plot separated by a line representing the threshold-value. This function will turn
# turn x.min() to threshold-value as a red rectangle to represent the thresholding-prediction of -1. It will then create
# threshold-value to x.max() as a green rectangle to represent the thresholding-prediction of 1. A dotted line will be
# colored to match which output value was being targeted by the user's b-value. The dotted line will be red if they
# selected b = -1 and green if they selected b = 1. The threshold-value will be plotted at (threshold, 0) and given
# either a red upside-down triangle pointing the user at a glance to the -1 outputs, or it will output a green
# right-side triangle pointing the user at a glance to the 1 outputs.
#
# Note: The user can input a different 1-d x and y arrays assuming the y-array contains either 1 or -1 values against
# the previously predicted threshold-value :)
class ThresholdLearner(object):

    # Initializes the class
    def __init__(self, minthreshold=0.5, maxthreshold=1.0, iteration=10, b=1):

        # used to hold the minimum threshold
        self.minthreshold = minthreshold

        # used to hold the threshold value
        self.threshold = self.minthreshold

        # used to hold the maximum threshold
        self.maxthreshold = maxthreshold

        # used to hold the amount of iterations
        self.iteration = iteration

        # used to determine refinement on the threshold
        self.increment = (self.maxthreshold - self.minthreshold) / self.iteration

        # used to determine if the user wants 1 or 0
        self.b = b

    # Takes in a 1-d array of in input values and a set of 1 or -1 for y-values and any real x-values
    def predict(self, x, y):

        # Number of misclassifications, creates an array to hold the number of misclassifications
        self.errors = []

        # find not b based off of b-value
        if self.b == 1:
            not_b = -1
        else:
            not_b = 1

        # loop though the x-values and determine what output they should get as a guess, then check
        for i in range(self.iteration):
            # loop through the copy of x and determine a predicted y value
            y_predict = np.where(x > self.threshold, self.b, not_b)

            # Create a comparison array
            if not_b == 1:
                compare = np.array([y, np.flipud(y_predict)])
            else:
                compare = np.array([y, y_predict])

            # Originally I thought I had to make a compare function but numpy is amazing
            # This is needs to be nerfed. I move to nerf Python for being too powerful
            error = np.sum(np.absolute(np.diff(compare, axis=0)))

            # add to the error list
            self.errors.append(error)

            # Check error to see if we found the correct threshold else iterate the threshold by the increment
            if error == 0:
                return self
            else:
                # increase the threshold
                self.threshold = self.minthreshold + self.increment * i

        # We need to find the best approximation by looking at which threshold had the least amount of error
        # We can use this trick from https://thispointer.com/numpy-amin-find-minimum-value-in-numpy-array-and-its-index/
        index = np.where(self.errors == np.amin(self.errors))

        # threshold will be minimum threshold + increment * index of least error
        self.threshold = self.minthreshold + self.increment * index[0].min()
        return self

    # Graph the threshold function give a 1-d x-array and y-array, where y contains only 1, or -1
    def graph(self, x, y):

        if self.b == 1:
            color = 'g^'
            scolor = 'green'
        else:
            color = 'rv'
            scolor = 'red'
        # I just used the matplotlib stuff and made this :) it was very fun and stuff
        fig, ax = plt.subplots()

        # make a scatter plot
        ax.scatter(x, y)

        # plot the point of the threshold
        ax.plot(self.threshold, 0, color, label=f'Threshold = {np.round(self.threshold, 4)}, b ={self.b}')

        # plots the lower square side
        plt.axvspan(round(np.min(x)), self.threshold, facecolor='r', alpha=0.5)

        # plots a doted line between the threshold value
        ax.vlines(self.threshold, -1, 1, colors=scolor, linestyles='dashed')

        # plots the upper square side
        plt.axvspan(self.threshold, (np.ceil(np.max(x))), facecolor='g', alpha=0.5)

        # add in graph title and axis titles
        plt.title('1-D Threshold Graph')
        plt.xlabel('User Inputted Data')
        plt.ylabel('Binary Output: 1 or -1')

        # plot the legend, so we can see the threshold value and show the user :)
        plt.legend()
        plt.show()


# This is a 1-D interval learner which uses a bruteforce algorithm of adjusting two different threshold values that
# that out put the best approximation of the dataset. To initialize use
# IntervalLearner(minthreshold, maxthreshold, iteration, b) and assign to an object. The following will explain
# what each of the variables do:
# minthreshold: is the user's lowest or starting threshold-value
# maxthreshold: is the user's highest or ending threshold-value
# iteration: is the user's number of times the threshold will increment through the threshold. The program will exit if
#            a perfect threshold-value is found before the number of iterations have been reached :)
# b: is the value either (1) or (-1) which will determine what type of desired binary data is being targeted
# Note: this program will auto calculate the precision of the increment using the following equation:
#       increment = (maxthreshold - minthreshold) / iteration
# If more precision is required, increase the number of iterations for a smaller threshold step :)
#
# Use obj.predict(x, y) to find the closest threshold that can represent a 1-D dataset given desired outputs are binary
# not (base2) but either 1, or -1
# x: should be a 1-d numpy array of float/double values
# y: should be 1-d numpy array of 1 or -1 (any integer value may be used in place of -1)
# obj.error: will give the list of errors calculated between each iteration
# obj.threshold: will give the user the predicted threshold-value
#
# Use graph(x, y) to get a scatter plot separated by two lines that show how the intervals represent the data
#
# Note: The user can input a different 1-d x and y arrays assuming the y-array contains either 1 or -1 values against
# the previously predicted threshold-value :)
# Note 2: I have not figured out why a certain dataset does not work correctly on my learner
class IntervalLearner(object):

    # Initializes the class
    def __init__(self, minthreshold=0.5, maxthreshold=1.0, iteration=10, b=1):
        # used to hold the minimum threshold
        self.minthreshold = minthreshold

        # used to hold the maximum threshold
        self.maxthreshold = maxthreshold

        # used to hold the threshold value first threshold value
        self.threshold1 = self.minthreshold

        # used to hold the threshold value for the second threshold value
        self.threshold2 = self.maxthreshold

        # used to hold the amount of iterations
        self.iteration = iteration

        # used to determine refinement on the threshold
        self.increment = (self.maxthreshold - self.minthreshold) / self.iteration

        # used to determine if the user wants 1 or 0
        self.b = b

    # Takes in a 1-d array of in input values and a set of 1 or -1 for y-values and any real x-values
    def predict(self, x, y):

        # Number of misclassifications, creates an array to hold the number of misclassifications
        self.errors = []

        # find not b based off of b-value
        if self.b == 1:
            not_b = -1
        else:
            not_b = 1
            self.threshold1 = self.maxthreshold
            self.threshold2 = self.minthreshold

        # loop though the x-values and determine what output they should get as a guess, then check
        for i in range(self.iteration):
            # loop through the copy of x and determine a predicted y-values
            if self.b == 1:
                # y = b : if min threshold < x < max threshold
                y_predict1 = np.where(x < self.threshold1, self.b, (np.where(x > self.threshold2, self.b, not_b)))
            else:
                # y = not(b) : if min threshold < x < max threshold
                y_predict1 = np.where(x < self.threshold1, not_b, (np.where(x > self.threshold2, not_b, self.b)))

            # loop through the copy of x and determine a predicted y-values
            # Create a comparison array
            if not_b == 1:
                compare = np.array([y, np.flipud(y_predict1)])
            else:
                compare = np.array([y, y_predict1])

            # Originally I thought I had to make a compare function but numpy is amazing
            # This is needs to be nerfed. I move to nerf Python for being too powerful
            error = np.sum(np.absolute(np.diff(compare, axis=0)))
            #print(self.errors)
            #print(y_predict1)
            # add to the error list
            self.errors.append(error)

            # Check error to see if we found the correct threshold else iterate the threshold by the increment
            if error == 0:
                return self
            else:
                if self.b == 1:
                    # increase the threshold1
                    self.threshold1 = self.minthreshold + self.increment * i
                    # decrease the threshold2
                    self.threshold2 = self.maxthreshold - self.increment * i
                else:
                    # increase the threshold2
                    self.threshold2 = self.minthreshold + self.increment * i
                    # decrease the threshold1
                    self.threshold1 = self.maxthreshold - self.increment * i

        # We need to find the best approximation by looking at which threshold had the least amount of error
        # We can use this trick from https://thispointer.com/numpy-amin-find-minimum-value-in-numpy-array-and-its-index/
        index1 = np.where(self.errors == np.amin(self.errors))
        index2 = np.where(self.errors == np.amin(self.errors))

        # threshold will be minimum threshold + increment * index of least error
        if self.b == 1:
            self.threshold1 = self.minthreshold + self.increment * index1[0].min()
            self.threshold2 = self.maxthreshold - self.increment * index2[0].min()
        else:
            self.threshold2 = self.minthreshold + self.increment * index1[0].min()
            self.threshold1 = self.maxthreshold - self.increment * index2[0].min()
        return self

    # Graph the threshold function give a 1-d x-array and y-array, where y contains only 1, or -1
    def graph(self, x, y):
        # Used to make Graph look pretty depending on the value of b
        if self.b == 1:
            color1 = 'g<'
            scolor1 = 'green'
            fcolor1 = 'g'
            color2 = 'r>'
            scolor2 = 'red'
            fcolor2 = 'r'
        else:
            color1 = 'r>'
            scolor1 = 'red'
            fcolor1 = 'r'
            color2 = 'g<'
            scolor2 = 'green'
            fcolor2 = 'g'
        # I just used the matplotlib stuff and made this :) it was very fun and stuff
        fig, ax = plt.subplots()

        # make a scatter plot
        ax.scatter(x, y)

        # plot the point of the threshold
        ax.plot(self.threshold1, 0, color1, label=f'Threshold = {np.round(self.threshold1, 4)}, b ={self.b}')
        ax.plot(self.threshold2, 0, color2, label=f'Threshold = {np.round(self.threshold2, 4)}, b = {self.b}')

        # plots the lower square side
        plt.axvspan(round(np.min(x)), self.threshold1, facecolor=fcolor1, alpha=0.5)
        # plots the upper square side
        plt.axvspan(self.threshold2, (np.ceil(np.max(x))), facecolor=fcolor2, alpha=0.5)
        # plots the middle square side
        plt.axvspan(self.threshold1, self.threshold2, facecolor='b', alpha=0.5)

        # plots a doted line between the threshold values
        ax.vlines(self.threshold1, -1, 1, colors=scolor1, linestyles='dashed')
        # plots a doted line between the threshold values
        ax.vlines(self.threshold2, -1, 1, colors=scolor2, linestyles='dashed')

        # add in graph title and axis titles
        plt.title('Interval Graph')
        plt.xlabel('User Inputted Data')
        plt.ylabel('Binary Output: 1 or -1')

        # plot the legend, so we can see the threshold value and show the user :)
        plt.legend()
        plt.show()


# This is a N-D logistic learner which uses a gradient batch decent to calculate the best logistic function that
# approximates the data set. Use obj = LogisticLearner(rate=float, niter=float) to create an instance of the class.
# the variable rate is used as a refining tool as it moves along the data. Use a smaller value if the output was not as
# refined as you want. The niter variable is used for the number of iterations that it spend to find the best function.
# X is a (n by m) numpy array and y is a 1-d by m array that holds either 1 or 0 as labels for the data set.
#
# Use obj.fit(X, y) to create a best logistic line for the data
# Use obj.accuracy(y) to find the accuracy of the logistic line, will return a percentage value
# Use obj.predict(X) to on a data set to see how the fitted logistic line to a dataset
# Use obj.graph(X, y) to see the outputs of the actual data and the predicted data
#
# Note: I was not able to get a logistic line to be made on the dataset, but I could not get four of plots to show up
# correctly
class LogisticLearner(object):

    # Initializes the class
    def __init__(self, rate=0.01, niter=100):
        self.rate = rate  # learning rate
        self.niter = niter  # number of iterations
        self.bias = 0 # default bias value

    # Create a model fitted to data
    def fit(self, X, y):
        # get the number of targets and features
        self.target, self.features = X.shape

        # create n-feature size weights
        self.weights = np.zeros((self.features, 1))

        # loop through and train
        for i in range(self.niter + 1):
            # calculate a rough y-predicted values using the sigmoid function
            self.y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)

            # calculate the change of the weights using gradient decent
            delta_w = self.deltaW(X, y, self.y_predict)

            # calculate the angle of the gradient of decent
            delta_b = self.deltaB(y, self.y_predict)

            # adjust the weights
            self.weights -= self.rate * delta_w

            # adjust the bias
            self.bias -= self.rate * delta_b

        return self

    # Used to find the predicted value from the true data set, compared to sklearn's model, which had this function
    def accuracy(self, y):
        # Add up the number of times that a value was calculated correctly divided by the total amount of values
        precent_error = np.sum(y == self.y_predict) / self.target
        return precent_error

    # Predict value based off the weights and bias from the fit, then return the values
    def predict(self, X):
        # Use the dot product of the inputted data and the weights from the fitted model + the bias then use the sigmoid
        # function to get the values of the predicted logistic function
        y_predicted = self.sigmoid(np.dot(X, self.weights) + self.bias)
        # Only use the values that are above 0.5 to select which label it belongs in
        self.y_predict = y_predicted > 0.5
        # Return the predicted labels
        return self.y_predict

    # Graph the learner
    def graph(self, x, y):

        # They showed how to add in a label for each line and add in test
        plt.subplot(211)
        plt.scatter(x[:, 0], x[:, 1], c=y, label=f'True y correlation = {1}')
        plt.title('Actual Values of the Data')
        plt.legend(loc='best')

        plt.subplot(212)
        plt.scatter(x[:, 0], x[:, 1], c=self.y_predict, label=f'Predicted y correlation = {np.round(self.accuracy(y),3)}')
        plt.title('Predicted Values using Logistic Regression')
        plt.legend(loc='best')

        # Add in the binary choice for both types
        #t = np.linspace(np.min(x[:, 0])-3.2, np.max(x[:, 0]), len(x[:, 0]))
        #slope = self.weights[0]
        #intercept = self.weights[1]
        #sig = 1/(1+np.exp(-(t*slope + intercept)))
        #plt.plot(t, sig, '-', color='orange')

        # create the legend and plot it to user
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
        plt.show()

    # Sigmoid function
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    # Cost function using gradient descent
    def cost(self, y, y_predict):
        return (-1 / self.target) * np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))

    # Change of the weights
    def deltaW(self, X, y, y_predicted):
        return np.dot(X.T, (y_predicted - y)) / self.target

    # Change of the bias
    def deltaB(self, y, y_predicted):
       return 1 / self.target * np.sum(y_predicted - y)


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


# SoftSVM class that uses linear algebra to calculate a line of best fit that attempts to split the data. To
# initialize please use: SoftSVM(rate = float, niter=float, lamda=float) and assign to an object.
# X is an array of real values in the shape of X:X[0]|X[1] and y is an array the same length of X[0] and must be
# filled with either 1 or -1. Use ndarrays for all the two vectors.
#
# Use obj.fit(X, y) where X is a (2 by n) dimensional numpy array and y is a (1 by n) dimensional numpy array using test
# test data to find a linear line between the data as well as the bounds of the dataset.
# Use obj.predict(X) to use the fitted line on a data set, where X is a (n by 2) dimensional numpy array. Returns a
# (n by 1) dimensional numpy array.
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


# Use plot_decision_regions(X = (2 by n), y = (1 by n) , classifier = classifier that has predict, resolution = view of
# data (float))
# I decided to just call this function my classes to make it easier on the user, but they can also call this fucntion
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # create a np array
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # I considered using on of the predefined color gradient, but I got too flustered, so
    # Thanks to "Max Kleiner"on stackoverflow.com who recommended this work around, which lowers the error level
    # https://stackoverflow.com/questions/55109716/c-argument-looks-like-a-single-numeric-rgb-or-rgba-sequence
    matplotlib_axes_logger.setLevel('ERROR')

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    # creates the the shaded regions of the data
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    # set min and max values of the given data set
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    # add in the legend, so the used knows what values are being compared
    plt.legend(loc='upper left')

    plt.show()