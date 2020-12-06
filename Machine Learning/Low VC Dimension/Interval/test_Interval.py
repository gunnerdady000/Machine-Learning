import numpy as np
from Interval import  IntervalLearner

def main():
    test_interval()

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

    if __name__ == "__main__":
        main()