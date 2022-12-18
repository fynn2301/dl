import numpy as np
import copy
from Layers.Helpers import compute_bn_gradients

from Layers.Base import BaseLayer
class BatchNormalization(BaseLayer):
    def __init__(self, chanels: tuple) -> None:
        """constructor

        Args:
            chanels (tuple): chanel size
        """
        super().__init__()
        self.chanels = chanels
        self.decay = 0.8
        self.trainable = True
        
        self._optimizer_weights = None
        self._optimizer_bias = None
  
        self.initialize()
        
    @property
    def optimizer(self):
        return self._optimizer_bias

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer_weights = copy.deepcopy(value)
        self._optimizer_bias = copy.deepcopy(value)

    @optimizer.deleter
    def optimizer(self):
        del self._optimizer_bias
        del self._optimizer_weights    

    def initialize(self) -> None:
        """Initialize the bias and weights
        """
        self.weights = np.ones((self.chanels))
        self.bias = np.zeros((self.chanels))
        
    def forward(self, input_tensor: np.array) -> np.array:
        """forward propagation

        Args:
            input_tensor (np.array): input

        Returns:
            np.array: output
        """
        finfo = np.finfo(float)
        # if the input comes as an image it neads to be flattened
        reverse = False
        
        self.shape = input_tensor.shape
        if len(input_tensor.shape) == 4:
            reverse = True
            input_tensor = self.reformat(input_tensor)
        self.input_tensor = input_tensor
        
        # calculate muh and sigma 
        if self.testing_phase:
            muh = self.decay * self.mean_tilde_k0 + (1-self.decay) * self.mean
            std_power_2 = self.decay * self.var_tilde_k0 + (1-self.decay) * self.var
            self.mean_tilde_k0 = muh
            self.var_tilde_k0 = std_power_2
            
        else:
            muh = np.mean(input_tensor, 0)
            std = np.std(input_tensor, 0)
        
            # calculate the x tilde tensor
            std_power_2 = np.multiply(std, std)
            
            # safe the mean and var for later
            self.mean = muh
            self.mean_tilde_k0 = muh
            self.var = std_power_2
            self.var_tilde_k0 = std_power_2
        
        input_tensor_tilde = np.subtract(input_tensor, muh) / np.sqrt(std_power_2 + finfo.eps)[:,None].T
        self.input_tensor_tilde = input_tensor_tilde
        
        # calculate y
        output_tensor = np.multiply(self.weights, input_tensor_tilde) + self.bias
        
        # reverse the flattening and transposing
        if reverse:
            output_tensor = self.reformat(output_tensor)
            
        return output_tensor
    
    def backward(self, error_tensor: np.array) -> np.array:
        """backward propagating

        Args:
            error_tensor (np.array): error tensor from successor layer

        Returns:
            np.array: error tensor for the previous layer
        """
        reverse = False
        if len(error_tensor.shape) == 4:
            reverse = True
            error_tensor = self.reformat(error_tensor)
            
        error_tensor_0 = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean, self.var)
        
        self.gradient_bias = np.sum(error_tensor, 0)
        self.gradient_weights = np.sum(np.multiply(error_tensor,self.input_tensor_tilde), 0)
        
        if self._optimizer_weights is not None and self._optimizer_bias is not None:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self.gradient_bias)
            self.weights = self._optimizer_weights.calculate_update(self.weights, self.gradient_weights)
        
        
        if reverse:
            error_tensor_0 = self.reformat(error_tensor_0)
        return error_tensor_0
    
    def reformat(self, tensor: np.array) -> np.array:
        """reformats from and to image

        Args:
            tensor (np.array): before reformating

        Returns:
            np.array: after reformatting
        """
        if len(tensor.shape) == 4:
            shape = tensor.shape
            tensor = np.reshape(tensor,(shape[0],shape[1],shape[2]*shape[3]))
            tensor = np.transpose(tensor, (0,2,1))
            output_tensor = np.reshape(tensor, (shape[0] * shape[2] * shape[3], shape[1]))
        
        elif len(tensor.shape) == 2:
            shape = self.shape
            tensor = np.reshape(tensor,(shape[0], shape[2] * shape[3], shape[1]))
            tensor = np.transpose(tensor, (0,2,1))
            output_tensor = np.reshape(tensor, shape)
        return output_tensor
