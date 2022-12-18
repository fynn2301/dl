import numpy as np
from Layers.Base import BaseLayer
class Sigmoid(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, input_tensor: np.array) -> np.array:
        """forward propagation

        Args:
            input_tensor (np.array): input

        Returns:
            np.array: output
        """
        output_tensor = 1 / (1+np.exp(input_tensor * -1))
        self.activations = output_tensor
        return output_tensor
    
    def backward(self, error_tensor: np.array) -> np.array:
        """backward propagation

        Args:
            error_tensor (np.array): successor error tensor

        Returns:
            np.array: previous error tensor
        """
        derivative = np.multiply(self.activations, 1 - self.activations)
        error_tensor_0 = np.multiply(error_tensor,derivative)
        return error_tensor_0