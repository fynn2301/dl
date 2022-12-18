import numpy as np
from Layers.Base import BaseLayer
class TanH(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, input_tensor: np.array) -> np.array:
        """forward propagation

        Args:
            input_tensor (np.array): input

        Returns:
            np.array: output
        """
        output_tensor = np.tanh(input_tensor)
        self.activations = output_tensor
        return output_tensor
    
    def backward(self, error_tensor: np.array) -> np.array:
        """backward propagation

        Args:
            error_tensor (np.array): successor error tensor

        Returns:
            np.array: previous error tensor
        """
        derivative = 1 - self.activations ** 2
        error_tensor_0 = np.multiply(error_tensor,derivative)
        return error_tensor_0