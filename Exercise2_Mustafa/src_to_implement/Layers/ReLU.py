from Layers.Base import BaseLayer
import numpy as np


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.gradient_tensor = None

    def forward(self, input_tensor):
        output_tensor = np.where(input_tensor > 0, input_tensor, 0)
        self.input_tensor = input_tensor
        return output_tensor

    def backward(self, error_tensor):
        prev_error_tensor = np.where(self.input_tensor > 0, error_tensor, 0)
        return prev_error_tensor

