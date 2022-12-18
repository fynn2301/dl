import numpy as np

from Layers import Base

class Optimizer():
    def  __init__(self) -> None:
        self.regularizer = None
    
    def add_regularizer(self, regularizer):
        """add a regularizer to the optimizer

        Args:
            regularizer (Class instanse): the regularizer for this optimizer
        """
        self.regularizer = regularizer

class Sgd(Optimizer):
    
    def __init__(self, learning_rate: float) -> None:
        """Constructor of the class Sgd

        Args:
            learning_rate (float): learning rate of the implemented neural network
        """
        self.learning_rate = learning_rate
        super().__init__()
    
    def calculate_update(self, weight_tensor: np.ndarray, gradient_tensor: np.ndarray) -> np.ndarray:
        """calculated the updated weights 

        Args:
            weight_tensor (np.ndarray): last weight
            gradient_tensor (np.ndarray): gradient

        Returns:
            np.ndarray: updated weight
        """
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        
        if self.regularizer != None:
            updated_weights -= self.regularizer.calculate_gradient(weight_tensor) * self.learning_rate
        
        return updated_weights
    
class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate: float, momentum_rate: float) -> None:
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        
        self.momentum = None
        super().__init__()
        
    def calculate_update(self, weight_tensor: np.array, gradient_tensor: np.array) -> np.array:
        """ Calculates and returns the updated weight tensor

        Args:
            weight_tensor (np.array): previous weight tensor
            gradient_tensor (np.array): gradient tensor

        Returns:
            np.array: updated weight tensor
        """
        if self.momentum is None:
            if type(gradient_tensor) == float:
                self.momentum = 0
            else:
                self.momentum = np.zeros(gradient_tensor.shape)
            
        self.momentum = self.momentum * self.momentum_rate - self.learning_rate * gradient_tensor
        updated_weights = weight_tensor + self.momentum
        
        if self.regularizer != None:
            updated_weights -= self.regularizer.calculate_gradient(weight_tensor) * self.learning_rate
            
        return updated_weights
      
class Adam(Optimizer):
    def __init__(self, learning_rate: float, mu: float, rho: float) -> None:
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.iteration = 1
        
        self.r_tensor = None
        self.v_tensor = None
        super().__init__()
        
    def calculate_update(self, weight_tensor: np.array, gradient_tensor: np.array) -> np.array:
        """ Calculates and returns the updated weight tensor

        Args:
            weight_tensor (np.array): previous weight tensor
            gradient_tensor (np.array): gradient tensor

        Returns:
            np.array: updated weight tensor
        """
        if self.r_tensor is None or self.v_tensor is None:
            if type(gradient_tensor) == float:
                self.r_tensor = 0
                self.v_tensor = 0
            else:
                self.r_tensor = np.zeros(gradient_tensor.shape)
                self.v_tensor = np.zeros(gradient_tensor.shape)
        
        self.v_tensor = self.mu * self.v_tensor + (1 - self.mu) * gradient_tensor
        self.r_tensor = self.rho * self.r_tensor + (1 - self.rho) * np.multiply(gradient_tensor, gradient_tensor)
        
        v_tensor_corrected = self.v_tensor / (1 - pow(self.mu, self.iteration))
        r_tensor_corrected = self.r_tensor / (1 - pow(self.rho, self.iteration))
        
        self.iteration += 1
        
        finfo = np.finfo(float)
        
        updated_weights = weight_tensor - self.learning_rate * np.divide(v_tensor_corrected, np.sqrt(r_tensor_corrected) + finfo.eps)
        
        if self.regularizer != None:
            updated_weights -= self.regularizer.calculate_gradient(weight_tensor) * self.learning_rate
        
        return updated_weights
        