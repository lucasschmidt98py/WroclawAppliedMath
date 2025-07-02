import numpy as np
from abc import ABC , abstractmethod
from typing import List

class GradientDescentOptimizer(ABC):
    def __init__(self,learning_rate : float = .01):
        self.learning_rate = learning_rate


    @abstractmethod
    def optimize():
        pass

class MGD(GradientDescentOptimizer):
    def __init__(self,
                 alpha: float = .01,
                 beta: float  = .9,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6):
        self.alpha = alpha
        self.beta = beta
        self.max_iterations = max_iterations
        self.tolerance = tolerance


    def optimize(self
                 ,F
                 ,GradF
                 ,X0: List[float]):
        X = np.array(X0, dtype=np.float64)
        V = 0*np.array(X0, dtype=np.float64)
        cost_history = []

        for _ in range(self.max_iterations):
            cost = F(X)
            cost_history.append((X.tolist(), cost))
            new_V = self.beta * V + (1-self.beta) * np.array(GradF(X))
            new_X = X - self.alpha * new_V
            if np.linalg.norm(new_X - X) < self.tolerance:
                break
            V = new_V
            X = new_X
        return X, cost_history

class NAG(GradientDescentOptimizer):
    def __init__(self,
                 alpha: float = .01,
                 beta: float = .09,
                 max_iterations = 1000,
                 tolerance: float = 1e-6):
        self.alpha = alpha
        self.beta = beta
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(self,
                F,
                GradF,
                X0: List[float]):
        X = np.array(X0, dtype=np.float64)
        V = 0*np.array(X0, dtype=np.float64)
        cost_history = []
        for _ in range(self.max_iterations):
            cost = F(X)
            cost_history.append((X.tolist(), cost))
            new_V = self.beta * V + np.array(GradF(X - self.alpha*self.beta*V))
            new_X = X - self.alpha * new_V
            if np.linalg.norm(new_X - X) < self.tolerance:
                break
            V = new_V
            X = new_X
        return X , cost_history

class ADG(GradientDescentOptimizer):
    def __init__(self,
                alpha: float = .9,
                epsilon: float = .0001,
                max_iterations = 1000,
                tolerance: float = 1e-6):
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(self,
                 F,
                 GradF,
                 X0:List[float]):
        X = np.array(X0, dtype=np.float64)
        G = 0*np.array(X0, dtype=np.float64)
        cost_history = []
        for _ in range(self.max_iterations):
            cost = F(X)
            cost_history.append((X.tolist(), cost))
            new_G = G + np.array( GradF(X) )**2
            new_X = X - self.alpha * np.array( GradF(X) ) / (np.sqrt(new_G + self.epsilon))
            if np.linalg.norm(new_X - X) < self.tolerance:
                break
            G = new_G
            X = new_X
        return X , cost_history

class RMS(GradientDescentOptimizer):
    def __init__(self,
                alpha: float = .01,
                beta: float = .09,
                epsilon: float = .01,
                max_iterations = 1000,
                tolerance: float = 1e-6):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def optimize(self,
                 F,
                 GradF,
                 X0:List[float]):
        X = np.array(X0, dtype=np.float64)
        E = 0*np.array(X0, dtype=np.float64)
        cost_history = []
        for _ in range(self.max_iterations):
            cost = F(X)
            cost_history.append((X.tolist(), cost))
            new_E = self.beta * E + (1-self.beta) * np.array( GradF(X) )
            new_X = X - self.alpha * np.array( GradF(X) ) / (np.sqrt(new_E + self.epsilon))
            if np.linalg.norm(new_X - X) < self.tolerance:
                break
            G = new_E
            X = new_X
        return X , cost_history

class ADAM(GradientDescentOptimizer):
    def __init__(self,
                alpha: float = .01,
                beta1: float = .9,
                beta2: float = .999,
                epsilon: float = .0001,
                max_iterations = 1000,
                tolerance: float = 1e-6):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    def optimize(self,
                 F,
                 GradF,
                 X0:List[float]):
        X = np.array(X0, dtype=np.float64)
        m = 0*np.array(X0, dtype=np.float64)
        v = 0*np.array(X0, dtype=np.float64)
        cost_history = []
        for _ in range(self.max_iterations):
            cost = F(X)
            cost_history.append((X.tolist(), cost))
            new_m = self.beta1 * m + (1-self.beta1) * np.array(GradF(X))
            new_v = self.beta2 * v + (1-self.beta2) * np.array(GradF(X))**2
            mhat = new_m / (1-self.beta1)
            vhat = new_v / (1-self.beta2)
            new_X = X - self.alpha * mhat / (np.sqrt(vhat) + self.epsilon )
            if np.linalg.norm(new_X - X) < self.tolerance:
                break
            m = new_m
            v = new_v
            X = new_X
        return X , cost_history
