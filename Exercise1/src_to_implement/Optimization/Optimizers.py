import numpy as np
# NOTE: DONE
class Sgd():
    
    def __init__(self, learning_rate: float) -> None:
        """Constructor of the class Sgd

        Args:
            learning_rate (float): learning rate of the implemented neural network
        """
        self.learning_rate = learning_rate
        pass
    
    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray) -> np.ndarray:
        """calculated the updated weights 

        Args:
            weight_tensor (np.ndarray): last weight
            gradient_tensor (np.ndarray): gradient

        Returns:
            np.ndarray: updated weight
        """
        updated_weight = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weight
    
    