### K Nearest Neighbor

## Introduction
The k Nearest Neighbor is a supervised learner that calulates the distance from a given point to all other k nearest neighbors. This is extremly accurate for data and does not make any assumptions about the dataset, but it comes at the cost of being more RAM hungry than Chrome. So, please be carful, you can quickly use up more RAM then there is in the universe. 

## Theory 
![](images/knntheory.PNG)

## Class Outline
The class has the following functions: 
 - i.   __init__()
 - ii.  fit()
 - iii. predict()
 - iv.  accuracy()
 -  v.  plot()

# __init__(k=integer)
This function creates an instance of the class and assigns it to an object. The user must input an integer for the number of nearest neighbors they would like to caclulate. 

# fit(X=[n by 2]ndarray, y=[n by 1]ndarray)
This makes a copy of the input data where X is a 2-dimensional array with n-elements and y is a 1-dimensional array with n-elements. This is part of the reason why the memory usage is so high. 

# predict(X=[n by 2]ndarray)
This function uses the fitted (training) data against another dataset or the same data set. This function finds the distance between the x values of both data sets. It then sorts through all of the distances and creates k nearest neighbor output values. it returns a ndarray. 

# accuracy(y_pred=[n by 1], y=[n by 1])
This function returns the accuracy of the predicted values compared to the original dataset. This value will be a percentage. 

# plot(X=[n by 2]ndarray, y=[n by 1]ndarray)
This function uses the plot_decision_regions function which means only datasets with [n by 2] maybe used.

## Usage 
obj = KNN(integer)

obj.fit(X, y)

obj.plot(X, y)

y_pred = obj.predict(X)

obj.accuracy(y_pred, y)

## Output
The images below are the outputs of the test file. 

Here is the output of k=1 using a 2 feature set from the Iris data set.

![](images/knnoutput1.PNG)

Next is the output of k=15 using a 3 feature set from the Iris data set.

![](images/knnoutput2.PNG)

Output of k=1 using a 3 feature set from the Iris data set with an added point. 

![](images/knnoutput3.PNG)

Output of SKLearn's model given k=1 and using a 3 feature set from the Iris data set.

![](images/knnoutput4.PNG)
