"""
This module implements the RidgeRegression class.

It solves the ridge regression problem with L2 regularization.
"""

import numpy as np


class RidgeRegression:
    """
    Ridge regression model using L2 regularization.

    Implements the closed-form solution for Ridge Regression based on the
    formula in "The Elements of Statistical Learning" (Eq. 3.47).
    """

    def __init__(self, alpha: float = 0.0):
        """
        Initialize the RidgeRegression model.

        Parameters:
        alpha (float): Regularization strength. Default is 0 (no regularization).
        """
        self.alpha = alpha

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RidgeRegression":
        """
        Fit the RidgeRegression model to the training data.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n, m).
        Y (np.ndarray): Target vector of shape (n).

        Returns:
        RidgeRegression: The fitted model.
        """
        #  X = np.array, shape = (n, m)
        #  Y = np.array, shape = (n)

        # Based on Eq. (3.47) from Section, 3.4.1. Ridge regression, "The Elemments of Statistical Learning", Hastie et. all. # noqa
        n, m = X.shape
        X_b = np.c_[np.ones((n, 1)), X]
        I = np.eye(m + 1)
        I[0, 0] = 0
        self.theta = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ Y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the given input data.

        Parameters:
        X (np.ndarray): Feature matrix of shape (k, m).

        Returns:
        np.ndarray: Predicted target values of shape (k).
        """
        # input:
        #  X = np.array, shape = (k, m)
        # returns:
        #  Y = array(f(X_1), ..., f(X_k))

        # Based on Eq. (3.44) from Section, 3.4.1. Ridge regression, "The Elemments of Statistical Learning", Hastie et. all. # noqa
        k, m = X.shape
        X_b = np.c_[np.ones((k, 1)), X]
        return X_b @ self.theta
