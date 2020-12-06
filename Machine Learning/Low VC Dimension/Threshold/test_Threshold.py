import pandas as pd, numpy as np
from Threshold import ThresholdLearner

def test_threshold():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    print("Creating a non-separable data-set, this will result in an approximate threshold-value for the dataset")
    y = df.iloc[25:75, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    x = df.iloc[25:75, 0]
    print("Creating a threshold learner with a b-value of 1 using non-separable dataset...")
    thresh_pos = ThresholdLearner(1.0, 7.0, 100, 1)
    thresh_pos.predict(x, y)
    print("Number of iterations out of 100: ", len(thresh_pos.errors))
    thresh_pos.graph(x, y)
    print("Creating a threshold learner with a b-value of -1 using a non-separable dataset...")
    thresh_neg = ThresholdLearner(1.0, 7.0, 100, -1)
    thresh_neg.predict(x, y)
    print("Number of iterations out of 100: ", len(thresh_neg.errors))
    thresh_neg.graph(x, y)

    print("Creating a separable data set...")
    y_2 = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
    x_2 = np.array([1.2, 1.1, 1.5, 1.6, 1.8, 5.1, 5.5, 5.6, 5.7, 5.8])
    print("Graphing the positive non-separable threshold-leaner on the separable dataset...")
    thresh_pos.graph(x_2, y_2)
    print("Graphing the negative non-separable threshold-learner on the separable data set...")
    thresh_neg.graph(x_2, y_2)

    print("Retraining the positive threshold learner on the separable data set...")
    thresh_pos.predict(x_2, y_2)
    print("Number of iterations out of 100: ", len(thresh_pos.errors))
    print("Iteration errors as follows:", thresh_pos.errors)
    thresh_pos.graph(x_2, y_2)
    print("Retraining the negative threshold learner on the separable data set...")
    thresh_neg.predict(x_2, y_2)
    print("Number of iterations out of 100: ", len(thresh_neg.errors))
    print("Iteration errors as follows:", thresh_neg.errors)
    thresh_neg.graph(x_2, y_2)