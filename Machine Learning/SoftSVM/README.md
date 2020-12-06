### Soft Support Vector Machine (Soft SVM)

## Introduction
The support vector machine either soft or hard is supposed to find a linear line that best fits the data set. While that is what the Perceptron and the Linear Regression leaners create a line of best, they also include two more supporting lines that are boudned by the nearest opposing label points. The hard SVM is more ridged than the soft as it must fine the two opposing points, while the soft SVM uses a value, lambda, that acts as a threshold value for the difference between the points. 

## Theory 


## Class Outline 
 The class has four functions: 
 - i.   __init__()
 - ii.  fit()
 - iii. predict()
 - iv.  plot()

# __init(rate=float, niter=integer, lamda=float)__
This function is used to create an instance of the class and assign it to an object. The following variables are used for the following:

- rate is a foat variable that controls the rate of change that leaner uses, which means using a smaller value will increase the precision of the best fit line. 
- niter is an integer that controls how many of the times the leaner will spend trying to learn the data set. 
- lamda is a float variable that controls the difference between the nearest opposing label points, using a super small value will make the soft SVM act as a hard SVM. 

# fit(X=[n by 2]ndarray, y=[n by 1]ndarray)
This function uses a double for-loop that finds not only the line of best fit using a set of weights and linear algebra, but also uses the lambda variable to find the supporting vectors. 

# predict(X=[n by 2]ndarray)
This function follows the same idea as the Perctron class's predict function. This is done by using the dot product of the input data and the weights from fitted model plus the intercept. This variable then returns either a 1 or a -1. 

# plot(X=[n by 2]ndarray, y=[n by 1]ndarray) 
This function uses the plot_decision_regions function which means you can only use a X ndarray of n by 2 and y is a ndarray of n by 1. 

## Usage 

obj = SoftSVM(float, integer, float)

obj.fit(X, y) following the same arrays mentioned above 

obj.plot(X, y) 

obj.predict(X)

## Output 

