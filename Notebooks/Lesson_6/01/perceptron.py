""" This module implements a perceptron. """

import numpy as np
from numpy.random import seed
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import rcParams

# Set the plot figure size
rcParams["figure.figsize"] = 10, 5


class Perceptron(object):
    """
    Perceptron Classifier.

    Parameters
    ----------------------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Passes (epochs) over the training set.

    Attributes
    -----------------------
    w_: 1d-array
        Weights afetr fitting.
    errors_: list
        Number of misclassification in every epoch.
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """

        :param X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where 'n_samples' is the numberof samples and 'n_features' is the number of features.
        :param y: {array-like}, shape = [n_samples]
            Target values.
        :return: self: object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        Calculate the net input.

        :param X:
        :return:
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        Calculate class label after unit step.

        :param X:
        :return:
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
