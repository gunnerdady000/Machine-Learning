import pandas as pd, numpy as np, matplotlib.pyplot as plt
from ML import Perceptron, plot_decision_regions

def main():
    test_perceptron()

def test_perceptron():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    print("Creating a two-feature data set")
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    print("Creating Perceptron")
    pn = Perceptron(0.1, 10)
    print("Perceptron created")
    pn.fit(X, y)
    print("Perceptron fitted")
    print("Error List")
    print(pn.errors)
    print("Weight vector")
    print(pn.weight)
    print("Using plot_decision_regions function")
    plot_decision_regions(X, y, pn, 0.02)
    pn.plot(X, y)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.title('Petal Length vs Sepal Length')

    print("Creating a three-feature data set")
    y = df.iloc[0:150, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:150, [0, 1, 2]].values
    print("Creating Perceptron")
    pn1 = Perceptron(0.1, 10)
    print("Perceptron created")
    pn1.fit(X, y)
    print("Perceptron fitted")
    print("Error List")
    print(pn1.errors)
    print("Weight vector")
    print(pn1.weight)

    print("Creating a perceptron that does not have enough iterations to learn the three-feature data set")
    pn2 = Perceptron(0.1, 4)
    print("Perceptron created")
    pn2.fit(X, y)
    print("Perceptron fitted")
    print("Error List")
    print(pn2.errors)
    print("Weight vector")
    print(pn2.weight)

if __name__ == "__main__":
    main()