from platform import release
import numpy as np
from Layers.Base import BaseLayer
from Layers.ReLU import ReLU
from Layers.SoftMax import SoftMax

class FullyConnected(BaseLayer):
    def __init__(self, input_size: int, output_size:int) -> None:
        """Instructor of the Fully Connected Layer

        Args:
            input_size (int): number of input values
            output_size (int): number of output vaulue
        """
        # initilize the inheritence
        super().__init__()
        
        # set the size of input an output
        self.input_size = input_size
        self.output_size = output_size
        
        # the layer should be trainable
        self.trainable = True
        
        # init the weights
        self.weights = np.random.rand(self.input_size + 1, self.output_size)
        
        # optimizer
        self._optimizer = None

    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @optimizer.deleter
    def optimizer(self):
        del self._optimizer

        
    
    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """This function calculates the forward propagation of the layer

        Args:
            input_tensor (np.ndarray): Input array of the layer

        Returns:
            np.ndarray: Output array of the Layer
        """
        # calculate the output
        batch_size = input_tensor.shape[0]
        input_tensor_bias = np.c_[input_tensor, np.ones(batch_size)]
        output_tensor = input_tensor_bias.dot(self.weights)
        
        # safe the input for backward later
        self.input_tensor = input_tensor
        return output_tensor


    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        """This function calculates the error tensor for the prvious layer

        Args:
            error_tensor (np.ndarray): error tensor of the successor layer

        Returns:
            np.ndarray: _description_
        """
        # calculate the error tensor of the previous layer
        error_tensor_0 = error_tensor.dot(self.weights[:-1,:].T)
        
        # calculate the gradient tensor of the wights
        gradient_tensor_weights = self.input_tensor.T.dot(error_tensor)
        gradient_tensor_biases = np.mean(error_tensor, 0)
        
        gradient_tensor = np.r_[gradient_tensor_weights, [gradient_tensor_biases]]
        self.gradient_weights = gradient_tensor
        if self.optimizer != None:
            self.weights = self.optimizer.calculate_update(self.weights, gradient_tensor)
        return error_tensor_0