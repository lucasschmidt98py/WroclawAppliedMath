import numpy as np


class Perceptron:
    """
    A simple implementation of the Perceptron learning algorithm.

    The Perceptron is a binary classifier that can learn linearly separable patterns.
    It implements the following logical operations:
    - AND gate
    - OR gate
    Note: Cannot learn XOR operation as it's not linearly separable.

    Attributes:
        learning_rate (float): Step size for weight updates (default: 0.1)
        n_iterations (int): Number of training iterations (default: 100)
        params (numpy.ndarray): Combined weights and bias parameters
    """

    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 100):
        """
        Initialize Perceptron model.

        Args:
            learning_rate: Learning rate for weight updates (between 0 and 1)
            n_iterations: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = np.array([])

    def activate(self, x: float) -> int:
        """
        Step function activation.

        Args:
            x: Input value

        Returns:
            1 if input >= 0, else 0
        """
        value = 1 if x >= 0 else 0
        return value

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the perceptron on input data.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Note:
            Updates params in-place by learning from training examples
        """

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features + 1)

        for _ in range(self.n_iterations):
            for xi, target in zip(X, y):
                linear_output = np.dot(xi, self.weights[:-1]) + self.weights[-1]
                prediction = self.activate(linear_output)
                update = self.learning_rate * (target - prediction)
                self.weights[:-1] += update * xi
                self.weights[-1] += update

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.

        Args:
            X: Input samples of shape (n_samples, n_features)

        Returns:
            Array of predicted labels (0 or 1) for each input sample
        """

        linear_output = np.dot(X, self.weights[:-1]) + self.weights[-1]
        return np.array([self.activate(x) for x in linear_output])
