import pandas as pd, numpy as np
from LinearRegression import LinearRegression

def main():
    test_linearRegression()

def test_linearRegression():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    print("Creating a weak positive correlation data set")
    y = df.iloc[0:50, 0]
    x = df.iloc[0:50, 1]
    print("Creating Linear-Regression objet...")
    linear = LinearRegression()
    print("Predicting line of best fit given a weak positive correlation...")
    linear.predict(x, y)
    print("Graphing weak positive correlation...")
    linear.graph(x, y)
    print("Creating a random strong negative correlation dataset...")
    x = np.random.randn(50)
    y = np.random.randn(1) - 2 * x
    print("Graphing weak positive correlation vs strong negative correlation dataset...")
    linear.graph(x, y)
    print("Retraining the linear regression leaner for a strong positive correlation dataset...")
    linear.predict(x, y)
    print("Graphing strong negative correlation...")
    linear.graph(x, y)
    print("Creating a weak negative correlation...")
    y = df.iloc[0:50, 0]
    x = df.iloc[0:50, 1]
    np.negative(x)
    print("Retraining the linear regression learner for a weak negative correlation dataset...")
    linear.predict(x, y)
    print("Graphing weak negative correlation...")
    linear.graph(x, y)
    print("Making a uniform dataset...")
    x = np.random.rand(50)
    print("Retraining the linear regression learner for a uniform distribution...")
    linear.predict(x, y)
    print("Graphing uniform distribution...")
    linear.graph(x, y)

if __name__ == "__main__":
    main()