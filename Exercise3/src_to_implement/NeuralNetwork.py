import copy
import numpy as np

class NeuralNetwork():
    def __init__(self, optimizer, weights_initializer, bias_initializer) -> None:
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        
        self.loss = []
        self.label_tensor = []
        self.prdiction_tensor = []

        self._phase = None
    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value

    @phase.deleter
    def phase(self):
        del self._phase
            
    def  forward(self) -> np.array:
        """Forwarding the input of the network through every layer

        Returns:
            np.array: output of the last layer
        """
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor
        
        regularization_loss = 0
        
        # all layers till the loss layer
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            
            # sumup regularization loss
            if self.optimizer.regularizer != None:
                regularization_loss += self.optimizer.regularizer.norm(layer.weights)
            
        self.prediction_tensor = input_tensor
        
        loss = self.loss_layer.forward(input_tensor, label_tensor)
        
        if self.optimizer.regularizer != None:
            loss += regularization_loss
        
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
            layer.initialize(self.weights_initializer, self.bias_initializer)
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