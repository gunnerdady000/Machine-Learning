import numpy as np
from LogisticRegression import LogisticLearner
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

def main():
    test_logistic()

def test_logistic():
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = (iris.target != 0) * 1
    y_sk = y
    y = y[:, np.newaxis]
    logic = LogisticLearner(0.01, 10000)
    logic.fit(x, y)
    y_predict = logic.predict(x)
    print(logic.weights)
    print(logic.bias)
    print(f"Accuracy: ", logic.accuracy(y))
    clf = LogisticRegression(random_state=0).fit(x, y_sk)
    clf.predict(x[:2, :])
    print(clf.score(x, y))
    logic.graph(x, y)

if __name__ == "__main__":
    main()
