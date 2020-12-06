import pandas as pd, numpy as np, matplotlib.pyplot as plt
from SoftSVM import SoftSVM
from sklearn import datasets


def main():
    test_svm()


def test_svm():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    print("Testing Soft SVM as Hard SVM....")
    magnitudeDirection = SoftSVM()
    magnitudeDirection.fit(X, y)
    print("Weights are: ", magnitudeDirection.weight)
    print("Intercept value is: ", magnitudeDirection.b)
    magnitudeDirection.plot(X, y)

    print("Using recommended SKLearn graph setup...")
    clf = SoftSVM()
    clf.fit(X, y)

    coolerPlot(X, y, clf)
    print("Using Soft SVM...")
    clf = SoftSVM(0.001, 1000, 0.01)
    clf.fit(X, y)
    coolerPlot(X, y, clf)

    print("Testing Blob Data Set")
    X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    y = np.where(y == 0, -1, 1)
    magnitudeDirection.fit(X, y)
    magnitudeDirection.plot(X, y)
    coolerPlot(X, y, magnitudeDirection)


### Used for SVM ###
def get_hyperplane_value(x, w, b, offset):
    return (-w[0] * x + b + offset) / w[1]

### Used for SVM ###
def coolerPlot(X, y, clf):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])
    x1_1 = get_hyperplane_value(x0_1, clf.weight, clf.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.weight, clf.b, 0)
    x1_1_m = get_hyperplane_value(x0_1, clf.weight, clf.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf.weight, clf.b, -1)
    x1_1_p = get_hyperplane_value(x0_1, clf.weight, clf.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, clf.weight, clf.b, 1)
    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k')
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k')
    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])
    plt.show()
    return
########################################################################################################################


if __name__ == "__main__":
    main()