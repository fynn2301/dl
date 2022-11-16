import numpy as np
from Layers.Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input_tensor: np.array) -> np.array:
        """Turns a n dim tensor into a 2d tensor

        Args:
            input_tensor (np.array): n dim input tensor

        Returns:
            np.array: transformed input tensor
        """
        # save the shape to reshape on backward
        self.last_shape = input_tensor.shape
        
        # flatten tensor and then reshape it to a 2d tensor
        input_flatten = input_tensor.flatten()
        batch_size = input_tensor.shape[0]
        output_tensor = np.reshape(input_flatten, (batch_size, -1))
        
        return output_tensor
    
    def backward(self, input_tensor: np.array) -> np.array:
        """Transforms the tensor back to a n dim tensor

        Args:
            input_tensor (np.array): 2d input tensor

        Returns:
            np.array: _description_
        """
        input_flatten = input_tensor.flatten()
        output_tensor = input_flatten.reshape(self.last_shape)
        return output_tensor
        