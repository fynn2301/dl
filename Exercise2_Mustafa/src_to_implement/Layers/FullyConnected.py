from Layers.Base import BaseLayer
from Layers.Initializers import UniformRandom
import numpy as np


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.rand(input_size + 1, output_size)

        self._optimizer = None

        self.input_tensor2 = None
        self.gradient_weights = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, new_optimizer):
        self._optimizer = new_optimizer

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        self.weights = np.append(self.weights, bias, axis=0)

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        input_tensor2 = np.c_[input_tensor, np.ones(batch_size)]
        output_tensor = np.dot(input_tensor2, self.weights)

        self.input_tensor2 = input_tensor2

        return output_tensor

    def backward(self, error_tensor):
        new_error_tensor = np.dot(error_tensor, self.weights.T)

        gradient_tensor = np.dot(self.input_tensor2.T, error_tensor)
        self.gradient_weights = gradient_tensor

        if self._optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, gradient_tensor)
        new_error_tensor = new_error_tensor[:, :-1]
        return new_error_tensor


if __name__ == "__main__":
    f = FullyConnected(5, 3)
    unif = UniformRandom()
    f.initialize(unif, unif)

