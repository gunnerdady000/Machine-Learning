import numpy as np
import matplotlib.pyplot as plt

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