import numpy as np
# NOTE: DONE

class CrossEntropyLoss():
    def __init__(self) -> None:
        pass
    
    def forward(self, prediction_tensor: np.array, label_tensor: np.array) -> np.array:
        """This function calculates the loss value according to the loss formula

        Args:
            prediction_tensor (np.array): prediction of the layer
            label_tensor (np.array): labels to the predition

        Returns:
            np.array: loss values
        """
        # safe the prediction tensor for backward
        self.last_prediction = prediction_tensor
        
        # get the indecies for the loss values
        indices_rows_array = np.arange(label_tensor.shape[0])
        label_indices_array = np.argmax(label_tensor, axis=1)
        
        # get the values of the index 
        prediction_values = prediction_tensor[indices_rows_array,label_indices_array]
        finfo = np.finfo(float)
        prediction_values = prediction_values + finfo.eps
        
        # calculate the loss
        loss = np.sum(-np.log(prediction_values))
        return loss
        
    def backward(self, label_tensor: np.array) -> np.array:
        """This function calculates the error_tensor for the previous layer

        Args:
            label_tensor (np.array): labels of the data

        Returns:
            np.array: error tensor for the previous layer
        """
        finfo = np.finfo(float) 
        error_tensor_0 = np.divide(-label_tensor, self.last_prediction + finfo.eps)
        return error_tensor_0