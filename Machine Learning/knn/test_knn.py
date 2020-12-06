import pandas as pd, numpy as np
from knn import KNN
from ML import plot_decision_regions
from sklearn import datasets, neighbors

def main():
    test_knn()

def test_knn():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    y = np.random.randint(2, size=100)
    x = df.iloc[0:100, [0, 2]].values
    print("Testing 2-D Iris data set with only one neighbor...")
    neighbor = KNN(k=1)
    neighbor.fit(x, y)
    neighbor.plot(x, y)
    print("Testing Iris data set with 15 neighbor...")
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    neighbor = KNN(15)
    neighbor.fit(x, y)
    y_pred = neighbor.predict(x)
    neighbor.accuracy(y_pred, y)
    neighbor.plot(x, y)
    print("Adding new point to dataset and testing with full Iris 1-k data set...")
    neighbor = KNN(1)
    neighbor.fit(x, y)
    y2 = np.array([1])
    y2 = np.append(y, y2)
    x2 = np.vstack([x, [5.0, 3.2]])
    neighbor.plot(x2, y2)
    print("Testing SKLearn's model...")
    clf = neighbors.KNeighborsClassifier(1)
    clf.fit(x, y)
    plot_decision_regions(x2, y2, clf)

if __name__ == "__main__":
    main()