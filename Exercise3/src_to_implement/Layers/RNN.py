from Layers.Base import BaseLayer
from Layers import FullyConnected, TanH, Sigmoid
import numpy as np
import copy
class RNN(BaseLayer):
    def __init__(self,input_size: int, hidden_size: int, output_size: int) -> None:
        """Constructor

        Args:
            input_size (int): denotes the dimension of the input vector 
            hidden_size (int): denotes the dimension of the hidden state
            output_size (int): denotes the dimension of the output
        """
        super().__init__()
        self.trainable = True
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.hidden_state = np.zeros((hidden_size))
        
        self.sub_layers = []
        
        self._optimizer0 = None
        self._optimizer1 = None
        
        self._memorize = False
        self.iteration = 0
        
        #gradients weights
        self._gradient_weights0 = None
        self._gradient_weights1 = None
         
        # sub layers
        self.fullyconnected_0 = FullyConnected.FullyConnected(input_size+hidden_size, hidden_size)
        self.fullyconnected_1 = FullyConnected.FullyConnected(hidden_size, output_size)
        self.tan_h = TanH.TanH()
        self.sigmoid = Sigmoid.Sigmoid()
        
        # hidden variable stack
        self.hidden_variable_forward = np.zeros((hidden_size))
        self.hidden_variable_backwards = np.zeros((hidden_size))
        
        # save the activations and input tensors and similar in a stack for the backwards pass
        self.input_state_stack = []
            
    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = copy.deepcopy(value)

    @memorize.deleter
    def memorize(self):
        del self._memorize
        
        
    @property
    def gradient_weights(self):
        return self._gradient_weights0
    
    @property
    def optimizer(self):
        return self._optimizer0

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer0 = copy.deepcopy(value)
        self._optimizer1 = copy.deepcopy(value)

    @optimizer.deleter
    def optimizer(self):
        del self._optimizer0
        del self._optimizer1
        
    @property
    def weights(self):
        return self.fullyconnected_0.weights

    @weights.setter
    def weights(self, value):
        self.fullyconnected_0.weights = copy.deepcopy(value)
        
    def initialize(self, weights_initializer, bias_initializer) -> None:
        """Initializes the weights and biases of the layer

        Args:
            weights_initializer (_type_): initializer for the weights
            bias_initializer (_type_): initializer for the biases
        """
        self.fullyconnected_0.initialize(weights_initializer, bias_initializer)
        self.fullyconnected_1.initialize(weights_initializer, bias_initializer)
        
    def forward(self, input_tensor: np.array) -> np.array:
        """forward propagation

        Args:
            input_tensor (np.array): input

        Returns:
            np.array: output
        """
        if not self.memorize:
            self.hidden_variable_forward = np.zeros((self.hidden_size))
        
        output_tensor = np.zeros((input_tensor.shape[0], self.output_size))
        
        for i, batch_tensor in enumerate(input_tensor):
            # get the input for the tan h layer
            x_tilde = np.concatenate((batch_tensor, self.hidden_variable_forward))
            x_tilde = np.expand_dims(x_tilde, axis=0)
            tan_h_input_tensor = self.fullyconnected_0.forward(x_tilde)
            
            # get the next hidden variable
            new_hidden_variable_forward = self.tan_h.forward(tan_h_input_tensor)
            self.hidden_variable_forward = new_hidden_variable_forward[0]
            
            # get the output
            sigmoid_input_tensor = self.fullyconnected_1.forward(new_hidden_variable_forward)
            output_tensor[i,:] = self.sigmoid.forward(sigmoid_input_tensor)
            
            # save the forward pass info to use later
            self.input_state_stack.append({"fullyconnected0": self.fullyconnected_0.input_tensor_bias,
                                           "tanh": self.tan_h.activations, 
                                           "fullyconnected1": self.fullyconnected_1.input_tensor_bias, 
                                           "sigmoid": self.sigmoid.activations})
        return output_tensor
    
    def backward(self, error_tensor) -> np.array:
        """_summary_

        Args:
            error_tensor (_type_): _description_

        Returns:
            np.array: _description_
        """
        batch_size = error_tensor.shape[0]
        if not self.memorize:
            self.hidden_variable_backwards = np.zeros((self.hidden_size))
        
        new_error_tensor = np.zeros((batch_size, self.input_size))
        
        for i, error_batch_tensor in enumerate(error_tensor[::-1]):
            # set the inputs before backwarding
            inputs = self.input_state_stack[batch_size - i - 1]
            self.fullyconnected_0.input_tensor_bias = inputs["fullyconnected0"]
            self.fullyconnected_1.input_tensor_bias = inputs["fullyconnected1"]
            self.tan_h.activations = inputs["tanh"]
            self.sigmoid.activations = inputs["sigmoid"]
            
            # run graph backwards
            sigmoid_back_out = self.sigmoid.backward(error_batch_tensor)
            tan_h_back_in = self.fullyconnected_1.backward(sigmoid_back_out)
            tan_h_back_in = tan_h_back_in[0]
            
            # add previous hidden error tensor and the backwards 
            new_tan_h_back_in = np.add(tan_h_back_in, self.hidden_variable_backwards)
            
            tan_h_back_out = self.tan_h.backward(new_tan_h_back_in)
            new_error_tensor_batch = self.fullyconnected_0.backward(tan_h_back_out)
            
            # save the hidden variables and the new error tensor
            self.hidden_variable_backwards = new_error_tensor_batch[0][self.input_size:]
            new_error_tensor[batch_size - i - 1,:] = new_error_tensor_batch[0][:self.input_size]
            
            #saving the gradient
            if i == 0:
                self._gradient_weights0 = self.fullyconnected_0.gradient_weights
                self._gradient_weights1 = self.fullyconnected_1.gradient_weights
            else:
                self._gradient_weights0 += self.fullyconnected_0.gradient_weights
                self._gradient_weights1 += self.fullyconnected_1.gradient_weights
            
        # update gradient
        if self._optimizer0 != None:
            self.fullyconnected_0.weights = self._optimizer0.calculate_update(self.fullyconnected_0.weights, self._gradient_weights0)
            self.fullyconnected_1.weights = self._optimizer1.calculate_update(self.fullyconnected_1.weights, self._gradient_weights1)
        
        return new_error_tensor
            
            
            
        