from Layers.Base import BaseLayer
import numpy as np


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output_tensor = None

    def forward(self, input_tensor):
        input_size = input_tensor.shape[1]

        max_per_batch = np.amax(input_tensor, axis=1)
        max_array = np.tile(max_per_batch, (input_size, 1)).T
        shift_input_tensor = input_tensor - max_array

        denominator_per_batch = np.sum(np.exp(shift_input_tensor), axis=1)
        denominator_array = np.tile(denominator_per_batch, (input_size, 1)).T
        output_tensor = np.divide(np.exp(shift_input_tensor), denominator_array)

        self.output_tensor = output_tensor

        return output_tensor

    def backward(self, error_tensor):
        subtrahend = np.multiply(error_tensor, self.output_tensor)
        subtrahend_sum = np.sum(subtrahend, axis=1)
        product = (error_tensor.T - subtrahend_sum).T

        prev_error_tensor = self.output_tensor * product

        return prev_error_tensor

