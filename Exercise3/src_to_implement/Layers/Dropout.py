import math
import numpy as np
from Layers.Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, propability: float) -> None:
        """constructor

        Args:
            propability (float): propability for the cells to get dropped
        """
        self.propability = propability
        super().__init__()
        
    def forward(self, input_tensor: np.array) -> np.array:
        """forward propagation

        Args:
            input_tensor (np.array): input

        Returns:
            np.array: output
        """
        if self.testing_phase:
            return input_tensor
        
        uniform_dist = np.random.uniform(size=input_tensor.shape)
        output_factor = np.where(uniform_dist < 0.25, 1, 0)
        
        self.last_output_factor = output_factor
        
        output = np.multiply(output_factor, input_tensor) * (1/self.propability)
        return output
        
    def backward(self, error_tensor: np.array) -> np.array:
        """error tensor

        Args:
            error_tensor (np.array): error tensor successot

        Returns:
            np.array: error tensor previous layer
        """ 
        if self.testing_phase:
            return error_tensor
        
        error_tensor_previous = np.multiply(error_tensor, self.last_output_factor) * (1/self.propability)
        return error_tensor_previous