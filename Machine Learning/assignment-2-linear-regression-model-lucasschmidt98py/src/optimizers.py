from typing import List

import numpy as np


class GradientDescentOptimizer:
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, tolerance: float = 1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(self, cost_function, gradient_function, initial_params: List[float]):
        """
        Performs gradient descent optimization.

        Args:
        cost_function: function that computes the cost
        cost_function: function that computes the gradient of the cost
        initial_params: Initial parameter values

        Returns:
        optimal_params: Optimized parameter values
        cost_history: List of cost values at each iteration (params, cost)
        """
        params = np.array(initial_params, dtype=np.float64)
        cost_history = []

        for iteration in range(self.max_iterations):
            cost = cost_function(params)
            gradient = np.array(gradient_function(params))
            cost_history.append((params.tolist(), cost))
            new_params = params - self.learning_rate * gradient
            if np.linalg.norm(new_params - params) < self.tolerance:
                break
            params = new_params

        return params, cost_history
