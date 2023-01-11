import copy
import numpy as np
from Layers.Base import BaseLayer


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
        self.gradient_weights = None
        
        # init the weights
        self.weights = np.random.rand(self.input_size + 1, self.output_size)
        
        # optimizer
        self._optimizer = None

    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = copy.deepcopy(value)

    @optimizer.deleter
    def optimizer(self):
        del self._optimizer

    def initialize(self, weights_initializer, bias_initializer) -> None:
        """Initializes the weights and biases of the layer

        Args:
            weights_initializer (_type_): initializer for the weights
            bias_initializer (_type_): initializer for the biases
        """
        weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        
        biases = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        
        self.weights = np.concatenate((weights, biases), axis=0)
    
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
        self.input_tensor_bias = input_tensor_bias
        return output_tensor


    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        """This function calculates the error tensor for the prvious layer

        Args:
            error_tensor (np.ndarray): error tensor[neuron_count x batch_size] of the successor layer

        Returns:
            np.ndarray: _description_
        """
        # calculate the error tensor of the previous layer
        error_tensor_0 = error_tensor.dot(self.weights.T)
        
        # calculate the gradient tensor of the wights
        gradient_tensor = self.input_tensor_bias.T.dot(error_tensor)
        
        self.gradient_weights = gradient_tensor
        if self.optimizer != None:
            self.weights = self.optimizer.calculate_update(self.weights, gradient_tensor)
        error_tensor_0 = error_tensor_0[:,:-1]
        return error_tensor_0