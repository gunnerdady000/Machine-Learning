import numpy as np
import matplotlib.pyplot as plt

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
