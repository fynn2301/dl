from Layers.Base import BaseLayer
import numpy as np
class RNN(BaseLayer):
    def __init__(self,input_size: int, hidden_size: int, output_size: int) -> None:
        """Constructor

        Args:
            input_size (int): denotes the dimension of the input vector 
            hidden_size (int): denotes the dimension of the hidden state
            output_size (int): denotes the dimension of the output
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.hidden_state = np.zeros((hidden_size))
        
        self.sub_layers = []
        
        super().__init__()
        
    def forward(self, input_tensor: np.array) -> np.array:
        """forward propagation

        Args:
            input_tensor (np.array): input

        Returns:
            np.array: output
        """
        x_tilde = np.concatenate(input_tensor, self.hidden_state)
        pass
        