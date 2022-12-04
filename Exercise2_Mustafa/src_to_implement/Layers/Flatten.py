import numpy as np
from Layers.Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        return np.reshape(input_tensor, (input_tensor.shape[0], input_tensor[0].size))

    def backward(self, error_tensor):
        return np.reshape(error_tensor, self.input_shape)

