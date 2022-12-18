from Layers.Base import BaseLayer
import numpy as np

class ReLU(BaseLayer):
    def __init__(self) -> None:
        """Constructor of the ReLU class
        """
        self.last_output = None
        # initilize the inheritence
        super().__init__()
    
    def forward(self, input_tensor: np.array) -> None:
        """This funtion calculates the ReLU function farward

        Args:
            input_tensor (np.array): input tensor of the funtion

        Returns:
            np.array: input tensor for the next layer
        """
        output_tensor = np.maximum(0, input_tensor)
        
        # save the output layer
        self.input_tensor = input_tensor
        
        return output_tensor
    
    def backward(self, error_tensor: np.array) -> np.array:
        """This funtion calculates the ReLU function backward

        Args:
            error_tensor (np.array): error tensor of the successor layer 

        Returns:
            np.array: error tensor of the previous layer
        """
        error_tensor0 = np.where(self.input_tensor > 0, error_tensor, 0)
        return error_tensor0