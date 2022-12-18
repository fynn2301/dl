from Layers.Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    def __init__(self) -> None:
        """Constructor of the class softmax
        """
        # initilize the inheritence
        super().__init__()
    
    def forward(self, input_tensor: np.array) -> np.array:
        """Calculates 

        Args:
            input_tensor (np.array): input tensor of the layer

        Returns:
            np.array: output tensor of the layer
        """
        var_size = input_tensor.shape[1]
        
        max_vect = np.asarray(np.amax(input_tensor,axis=1))
        max_array = np.tile(max_vect, (var_size, 1)).T
        
        input_tensor_shifted = input_tensor - max_array
        
        
        exp_sum_vect = np.sum(np.exp(input_tensor_shifted),axis=1)
        exp_sum_array = np.tile(exp_sum_vect, (var_size, 1)).T
        output_tensor = np.divide(np.exp(input_tensor_shifted),exp_sum_array)
        
        self.last_prediction = output_tensor
        return output_tensor
        
        
    def backward(self, error_tensor: np.array) -> np.array:
        """Calculates 

        Args:
            error_tensor (np.array): error_tensor of the successor layer

        Returns:
            np.array: error tensor passed to the previous layer
        """
        batch_size, neuron_size = error_tensor.shape
        
        multiplied = np.multiply(error_tensor, self.last_prediction)
        multiplied_sum = np.sum(multiplied, 1)
        
        subtracted = (error_tensor.T - multiplied_sum).T
                                
        error_tensor_0 = self.last_prediction * subtracted
        return error_tensor_0