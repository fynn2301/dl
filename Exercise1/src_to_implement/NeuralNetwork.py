import copy
import numpy as np

class NeuralNetwork():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        
        self.loss = []
        self.label_tensor = []
        self.prdiction_tensor = []
        
    def  forward(self) -> np.array:
        """Forwarding the input of the network through every layer

        Returns:
            np.array: output of the last layer
        """
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor
        
        # all layers till the loss layer
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            
        self.prediction_tensor = input_tensor
        
        loss = self.loss_layer.forward(input_tensor, label_tensor)
        self.loss.append(loss)
        
        return loss
    
    def backward(self) -> np.array:
        """Backward int error tensor from loss layer to init layer

        Returns:
            np.array: _description_
        """
        error_tensor = self.loss_layer.backward(self.label_tensor)
        
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
            
        return error_tensor
    
    def append_layer(self, layer) ->  None:
        """Appends layer to the network

        Args:
            layer (class): The layer that should be added to the network
        """
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        
        self.layers.append(layer)
        
    def train(self, iterations: int) -> None:
        """Train the network for 'iterations' iterations and save the the loss of every iteration in loss array

        Args:
            iterations (int): _description_
        """
        for _ in range(iterations):
            self.forward()
            self.backward()
        
    def test(self, input_tensor: np.array) -> np.array:
        """Test the network performance for thr input tensor

        Args:
            input_tensor (np.array): _description_

        Returns:
            np.array: _description_
        """
        # all layers till the loss layer
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor