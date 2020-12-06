import numpy as np
from math import sqrt
import matplotlib.pyplot as plt


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