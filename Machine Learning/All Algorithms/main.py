import pandas as pd, numpy as np, matplotlib.pyplot as plt
from ML import Perceptron, LinearRegression, ThresholdLearner, IntervalLearner, plot_decision_regions, LogisticLearner, \
    KNN, SoftSVM
from sklearn import datasets, neighbors
from sklearn.linear_model import LogisticRegression


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


def test_logistic():
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = (iris.target != 0) * 1
    y = y[:, np.newaxis]
    logic = LogisticLearner(0.01, 10000)
    logic.fit(x, y)
    y_predict = logic.predict(x)
    print(logic.weights)
    print(logic.bias)
    print(f"Accuracy: {np.sum(y == y_predict) / x.shape[0]}")
    clf = LogisticRegression(random_state=0).fit(x, y)
    clf.predict(x[:2, :])
    print(clf.score(x, y))
    logic.graph(x, y)


def test_interval():
    print("Creating a threshold learner with a b-value of 1 using a separable dataset...")
    val_pos = IntervalLearner(1.0, 6.0, 10, 1)
    print("Creating a threshold learner with a b-value of -1 using a separable dataset...")
    val_neg = IntervalLearner(1.0, 6.0, 10, -1)

    print("Creating a separable data set...")
    y_2 = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
    x_2 = np.array([1.2, 1.1, 1.5, 1.6, 1.8, 3.1, 3.2, 3.6, 3.7, 3.8, 5.1, 5.5, 5.6, 5.7, 5.8])
    # y_2 = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    print("length of y_2: ", len(y_2))
    print("lenght of x_2: ", len(x_2))
    print("Retraining the positive threshold learner on the separable data set...")
    val_pos.predict(x_2, y_2)
    print("Number of iterations out of 100: ", len(val_pos.errors))
    print("Iteration errors as follows:", val_pos.errors)
    val_pos.graph(x_2, y_2)
    print("Retraining the negative threshold learner on the separable data set...")
    # y_2 = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    val_neg.predict(x_2, y_2)
    print("Number of iterations out of 100: ", len(val_neg.errors))
    print("Iteration errors as follows:", val_neg.errors)
    val_neg.graph(x_2, y_2)


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


if __name__ == "__main__":
    main()
