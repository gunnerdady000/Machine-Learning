### Perceptron Class

## Introduction
The Perceptron Class is a supervised learner, as in it uses a set of biases (weights) to work itself
into the right answer, or within a certain amount error. The Perceptron Class works by taking a data and
that has been split into a 0 or 1, or binary set, as well as a training set of data. It uses a linear function and
its weights to slowly learn how to predict an output.

## Theory Time 
The Perceptron Class is a linear, binary classifier which uses matrix math (in this case) to figure out a linear line for a given dataset. 
The output of the Perceptron is either a 1 if w * x + b > 0 or it is a 0. 

w is the wieght vector of and acts as the slope of the line. 
b is the intercept of the line (stored within the weights matrix) 

The weights are adjusted using by using the current weight plus the learning rate times the rate of change between the current predicted output value and the actual output value plus the input values.

## Class Outline 
 The class has five functions: 
 - i.   __init__()
 - ii.  fit()
 - iii. net_input()
 - iv.  predict()
 -  v.   plot()

# __init(rate=float, niter=integer)__
This function creates an instance of the class.
The variable rate is used as a refining variable as it will change the step size as the fit() function tries to find the line of best fit.
The variable niter is used as how many maximum iterations the fitting function will use.
Adjust either the learning rate variable and or the number of iterations if the line does not fit the data.
Use: Percpetron(rate = float, niter=float) and assign to an object.

# fit(X=[2 by n] ndarray, y=[1 by n] ndarray)
This function takes in
a training vector that is composed of samples and features. The samples is just raw training data that it will
use to work on improving itself. The features is how many different types of training is within the data set.
Lastly, this function takes in a list of target values, these are the correct responses to the data set that the
Perceptron Class should achieve.
The Perceptron Class uses an array, specially a ndarray from the Numpy library, whose size is that of the
amount of features plus one, which is due to how the algorithm works. The first index is the Basis of the
entire Perceptron, the other two weights are meant to help the Perceptron bring balance to the world. One
could hypothetically create multiple instances of the Perceptron Class and then feed their outputs into the
inputs of another layer of Perceptron’s, but such an idea would never work… Anyway, some people prefer
to use a completely random bias value, but we did not do that for our class. We decided to loop through the
training data collecting any mistakes it made into a list that holds the amount of errors the Perceptron Class 
makes on its way to learning the training data. As the Perceptron is biased by the learning rate times the
delta between the target data and it’s attempt at predicting the rate, this value will be stored in the bias
index of the weight array, but this value will also be used for the weights themselves.
The Perceptron Class will form its weights, or inputs, by taking the bias and multiplying it by the current
raw data itself and add itself its previous value. If the bias is not equal to zero, then we start a counter and
add one to itself. After going through all of the data we will then add the error to the error list. Basically
this is how computer scientists treat machines as children, this method is the same as watching a kid fall
down a bunch times while trying to ride a bike until they figure it out. Of course, we do not want a kid to
keep proving a million times that they know how to ride a bike on their own, so we also need to check and
see if the Perceptron has mastered the data set. To solve this problem we decided that the Perceptron Class
has mastered the data set if it has two runs with a bias change of zero. If this is the case then we return the
fully trained Perceptron Class.

#predict(X[2 by n] ndarray])
The predict fucntion works by calling the net_input() function
given the training data, then determines if that result is greater than equal to zero. It then returns one if it is
true or it selects the negative one if the expression is false. The net_input() function utilizes the magic of
the Numpy library and the power of a dot product. This function simply takes in part of the data set, then
preforms the dot product between the data and its weights. It then adds this result with bias.

# plot(X[2 by n] ndarray, y[1 by n] ndarray)
This function calls the plot_decision_regions() fucntion which creates a graph of the data. 
The arrays must be the specified ndarrays and cannot exceed 5 features, unless you care to edit the plot_decision_regions() function.

## Usage
Just run the included test file or do the following: 
To run the Perceptron Class, the user must download the ML.py file into a folder that they are planning on
using the Perceptron Class. The user must then provide a training vector X, whose shape takes the form of [
number of samples, number of features], a target vector Y, whose values must either be -1 or 1, and be the
same length as the number of samples within the X vector. For our first example we will use the Iris data
set, given the following URL link 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'.

We will import the following libraries pandas, numpy, and matplotlib.

Next we will gather the data using the pandas library function read_csv(), which allows the data set to be
read from a csv file. We then create a list from the 4th column of the Irsi data set, by using y =
df.iloc[0:150, 4].values. We then need to change the values of the desired output, which in this case is the
y list values that match the Iris-setosa label and assign it -1 otherwise it becomes a 1, by using y =
np.wher(y == ‘Iris-setosa’, -1, 1). Lastly, we need the raw data input, which we will grab from the zero
column and the second column of the data set by using X = df.iloc[0:150, [0,2]].values.

Now we will need to import from the ML.py file in which the Perceptron Class exists by using from ML
import Perceptron. We will then create an object to become the almighty binary overlord that the
Perceptron Class is by using pn = Perceptron(0.1, 10). As stated before the values of the Perceptron are as
the learning rate and number of iterations that the Perceptron will use to learn the dataset. Lastly, we will
use the object and train it on the data sets by using pn.fit(X, y).

Lastly, just call the plot fucntion with the same dataset or a dataset that fits the previous requirements for the class.

Please see the included PDF for more documentation. 
