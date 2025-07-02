"""
This module implements the LinearRegression class.

It solves the least squares problem for linear regression.
"""

import numpy as np


class LinearRegression:
    """
    Linear regression model using the least squares method.

    Implements the closed-form solution for Linear Regression based on
    "The Elements of Statistical Learning", Hastie et. al.
    """

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "LinearRegression":
        """
        Fit the linear regression model to the data.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n, m).
        Y (np.ndarray): Target vector of shape (n).

        Returns:
        LinearRegression: Fitted model.
        """
        # input:
        #  X = np.array, shape = (n, m)
        #  Y = np.array, shape = (n)

        # Implement Eq. (3.6) from Chapter, 3.2. Linear Regression Models and Least Squares, "The Elemments of Statistical Learning", Hastie et. all. # noqa
        # Note: before applying the formula to X one should append to X a column with ones.
        n, m = X.shape
        X_b = np.c_[np.ones((n, 1)), X]
        self.theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values using the linear regression model.

        Parameters:
        X (np.ndarray): Feature matrix of shape (k, m).

        Returns:
        np.ndarray: Predicted target values of shape (k).
        """
        # input:
        #  X = np.array, shape = (k, m)
        # returns:
        #  Y = np.array(f(X_1), ..., f(X_k))

        # Implement Eq. (3.7) from Chapter, 3.2. Linear Regression Models and Least Squares, "The Elemments of Statistical Learning", Hastie et. all. # noqa
        k, m = X.shape
        X_b = np.c_[np.ones((k, 1)), X]
        Y_pred = X_b @ self.theta

        return Y_pred
