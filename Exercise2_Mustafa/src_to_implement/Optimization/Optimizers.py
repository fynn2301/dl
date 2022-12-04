import numpy as np


class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor1 = weight_tensor - self.learning_rate * gradient_tensor
        return weight_tensor1


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity = 0 if type(gradient_tensor) != np.ndarray else np.zeros(gradient_tensor.shape)
        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
        weight_tensor1 = weight_tensor + self.velocity
        return weight_tensor1


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho

        self.velocity = None
        self.alr = 0
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity = 0 if type(gradient_tensor) != np.ndarray else np.zeros(gradient_tensor.shape)

        self.velocity = self.mu * self.velocity + (1 - self.mu) * gradient_tensor
        self.alr = self.rho * self.alr + np.dot((1 - self.rho) * gradient_tensor, gradient_tensor)

        corrected_velocity = self.velocity / (1 - self.mu**self.k)
        corrected_alr = self.alr / (1 - self.rho**self.k)

        weight_tensor1 = weight_tensor - self.learning_rate * (corrected_velocity / (np.sqrt(corrected_alr) + np.finfo(float).eps))
        self.k += 1

        return weight_tensor1

