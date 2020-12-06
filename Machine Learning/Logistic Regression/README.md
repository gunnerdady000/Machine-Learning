### Logistic Regression

## Introduction 
The Logistic Regression is yet annother supervised, binary learner, as it makes a sigmoid fucntion to best approximate the given data set. The sigmoid fucntion creates an S-shaped line that seperates the data from each other as the values located above the line are given a value of 1 and the other values are given a value of -1. 

Thanks to user 
Artificial Intelligence - All in One from  Youtube for his reccomened math of doing logistic regression. I would have done this without SGD, which had poor perfromance when I tried just plotting logistic lines. 

https://www.youtube.com/watch?v=TTdcc21Ko9A&list=PLNeKWBMsAzboR8vvhnlanxCNr2V7ITuxy&index=5

## Theory  


## Class Overview 
The class has the following functions: 
 - i.    __init__()
 - ii.   fit()
 - iii.  sigmoid()
 - iv.   accuracy()
 - v.    deltaW()
 - vi.   deltaB()
 - vii.  predict()
 - viii. cost() Currrently, not used as I used it compare against SKLearn's Logistic learner
 - ix.   graph()
 
# __init__(rate=float, niter=float)
This function creates the class and assigns it to an object. The rate variable is a flot that determines how refined the leaner will increase its step size; smaller value yeilds more accuarte values. The niter value is used to hold the amount of iterations the program will run for as opposed to the maximum amount of time as with the linear regression learner. 

# fit(X=[2  by n]ndarray, y=[1 by n]ndarray of either 1's or -1's)
This function is a giant for loop that uses Stochastic Gradient Descent (SGD) to create a line faster and more effcient than if we just tried plotting and testing the output of different sigmoid lines on the data. 

# sigmoid(Z=[2  by n]ndarray)
This function preforms the sigmoid equation on a 2 by n ndarray and then returns those predicted output values. 

#accuracy(y=[1 by n]ndarray of either 1's or -1's)
This function returns the how accurate the predicted logistic regression model is to the actual values of the dataset in a percentage.

# deltaW(X=[2  by n]ndarray, y=[1 by n]ndarray of either 1's or -1's, y_predicted=[1 by n]ndarray of either 1's or -1's) 
This function calculates the dot product between the transpose of X and the predicted y value minus the y value. It then divides this number by total amount of target values and returns the difference of weights.

# deltaB(y=[1 by n]ndarray of either 1's or -1's, y_predicted=[1 by n]ndarray of either 1's or -1's) 
This fucntion returns the one divided by the number of target values times the summation of the predicted y value minus the target y value. 

# predict(X=[2  by n]ndarray)
This function is used to predict a logistic regression line that is clostest to the trained weights from the training data set. This uses the sigmoid function given the dot prodcut of the x-vector and the weights plus the bias value. It then uses 1/2 as a threshold value to determine it the output value should be 1 or -1

# cost(y=[1 by n]ndarray of either 1's or -1's, y_predict=[1 by n]ndarray of either 1's or -1's)
This function returns the cost of the SGD by comparing the neagtive inverse of the total target outputs times the summation of the current y-value times the log of the predicted y value plus (1 minus y) times the log of 1 - minus y. This was used to compare against the SKLearner's, which still smoked my implementation because they use witch craft. 

# graph(X=[2  by n]ndarray, y=[1 by n]ndarray of either 1's or -1's)
This function graphs the output of the data set as well as the precited outputs. I tried to copy SkLearn's test model page and get the logistic line printed but it did not work out, so this is the best I can do.

## Usage 
obj = LogisticLearner(float, float)
obj.fit(X, y)
obj.graph(x, y)

## Test File Output 
