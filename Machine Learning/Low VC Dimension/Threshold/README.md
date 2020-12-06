### Threshold Learner

## Introduction
This creates a line (threshold value) on a dimensional and all values that lie left the line are assinged a target output value or all. Values that lie to the right of the line are then given negative value. This is a supervised learner as well as a binary learner. This means the target vector must be either 1 or -1. The learner will adjust the line to the right or left until learner runs out of iteratinons or has mastered it. 

## Theory 
![](images/interval_theory.PNG)

## Class Outline 
The class is composed of the following functions:
- i.   __init__()
- ii.  predict() 
- iii. graph()

# __init__(minthreshold=float, maxthreshold=float, iteration=integer, b=integer[must only be 1 or -1])
The __init__() function initializes the class and assigns it to an object. The user must pass in the following values for class variables: 

minthreshold – float variable that holds the minimum threshold value, default is 0.5, but if the user does not know the minimum value in there set then it is recommend that they enter minimum value of their x-array values. 

maxthreshold – float variable that holds the maximum value that the threshold could be, default is 1.0, but if the user does not know the maximum value then it is recommended that they enter the maximum value of their x-array values. 

iteration – integer variable that holds the maximum amount of iterations the threshold learner will take to figure out the data set. Note: Increase this value for a more refined search of the most efficient   
b – integer (should be treated as a Boolean) used by the user to focus on either the desired y-output of 1, or -1, default is 1. Note: User can enter a different value then -1, as the program will still treat that other value as -1. 

Next, we need to talk about the threshold value, which is defaulted to the minimum threshold value once the user uses the class to predict a set of data. Lastly, we need to talk about the increment value, which takes the maximum threshold value and subtracts it from the minimum threshold value, this is then divided by the number of iterations that the user inputs. Note: This is why we informed the user to increase the number of iterations if they need a smaller step size between each iteration.   

# predict(x=[1 by n] ndarray, y[1 by n] ndarray) 
The predict() function takes in two 1-d numpy_array’s that hold the x-array and the y-array, which the y-array must be integers that are either 1, or -1. We decided to keep track of how many errors occur while trying to figure out the correct threshold value. We then determine if the user put in 1 or not 1 into the b-value, by checking to see if it is equal to 1. If it is equal to 1, the not_b is set to -1, otherwise not_b is set to 1. 

The function then loops from zero until the maximum value of iterations and preforms the following actions. We can leverage the numpy library by using the where() function, which will loop through the array and preform a regular expression on the entire array. As shown in the figure below, we look to see if the x-value is greater than the threshold value and if it is then we will assign a predicted y-value to the b-value, otherwise we assign the not_b value. 

We need to compensate for the ability of having a different b-value then just 1, so if not_b is equal to 1 then we need to set the 2-d compare array equal to the y-array in first row, followed by the middle-flipped predicted array, otherwise we do not flip the predicted array. The error usually would require another loop to be made, but numpy is a broken library, so we can make an entire function in a single line.

First, we will compare the difference between the two arrays, that have been stored within the 2-d compare array. We use “axis=0” as that will be between the rows within the library function. Now, we have an array of values that are either 1, 0, or -1, so we need to take the absolute value of the negative values by using absolute(). Now, we can use the same sum() function used for the Linear Regression Class. We have now looped through and added up all the errors that the threshold miscalculated. All in a single line instead of a function. 

Next, we add the newly calculated to the error array. We then check to see if we have zero error’s which means we have selected the correct threshold value and we can return the class. Otherwise, we will set the threshold value equal to the minimum threshold value plus the increment value times the current iteration.

Lastly, if we need to make sure that we use the threshold value with the least amount of miscalculations. This is done by finding the indices of where the minimum error occurred as they are directly correlated to the iteration value. Using the where() library function comparing the error array to another library function amin() which searches the error array and creates a new array that contains a list of indices that have the minimum values. This was shown on thispointer.com as someone else wondered if this was possible. We then can set the best-fit, not perfect-fit, threshold value equal to the minimum threshold value plus the increment times the new array holding the minimum error indices given the first value within that list. 

# graph(x=[1 by n] ndarray, y[1 by n] ndarray)
The graph() function takes in two 1-d numpy_array’s that hold the x-array and the y-array being integers that act as binary (1 or -1). Depending on what the user set the b-value to, will determine what color and directional arrow will be used to represent the threshold value. A green upwards arrow will be used to help the user see that the threshold was set with b-value = 1, as it points to all the y-values that are equal to 1. A red downward arrow will be assigned to help the user see that the threshold was set to reach b-value = 0, as it points to all the y-values that are equal to 0. We then make subplots as this was recommend by the matplotlib website for using some of their functions. 

We then make a scatter plot of the x-array and y-array, so the user can visualize their data. We then create plot the threshold value given y = 0, so it lies in the middle of the data. We also added a label that will show the user the threshold value as well as the b-value was set just in case the threshold arrow was not clear enough. 

Next, we create squares that represent either side of the threshold, with one side being red and the other being green. Again, the red side is supposed to show what values the threshold considers to be -1, while the green side shows what the threshold considers to be 1. The boundary of the red side is from the minimum x-value up to the threshold value and covers all of the y-axis. The boundary of the green side is from the threshold up to the maximum x-value and covers all of the y-axis. We then add in a vertical line at the x-axis value that the threshold is set at, which creates a dotted line from -1 to 1 on the y-axis. The color of this line will match the color of the threshold, as another visual aid to the user. Lastly, we added in labels and a title to help the user understand their dataset and what the threshold classifier is doing.   

## Usage
Use the inlcuded test file or use the following:

obj = LinearRegression()

obj.predict(x, y)

obj.graph(x, y)
