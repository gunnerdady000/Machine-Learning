import numpy as np
import matplotlib.pyplot as plt


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