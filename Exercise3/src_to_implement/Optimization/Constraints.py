import numpy as np

class L1_Regularizer():
    def __init__(self, alpha: float) -> None:
        """constructor

        Args:
            alpha (float): regularization weight (lambda)
        """
        self.alpha = alpha
        pass
    
    def calculate_gradient(self, weights: np.array) -> np.array:
        """calculates a (sub-)gradient on the weights needed for the optimizer

        Args:
            weights (np.array): weights

        Returns:
            np.array: (sub-)gradient
        """
        signs = np.sign(weights)
        output = signs * self.alpha
        return output
        
    def norm(self, weigths: np.array) -> float:
        """calculates the norm enhanced loss

        Args:
            weigths (np.array): weights

        Returns:
            float: loss
        """
        weigths_abs = np.abs(weigths)
        output = np.sum(weigths_abs) * self.alpha
        return output
    
class L2_Regularizer():
    def __init__(self, alpha: float) -> None:
        """constructor

        Args:
            alpha (float): regularization weight (lambda)
        """
        self.alpha = alpha
    
    def calculate_gradient(self, weights: np.array) -> np.array:
        """calculates a (sub-)gradient on the weights needed for the optimizer

        Args:
            weights (np.array): weights

        Returns:
            np.array: (sub-)gradient
        """
        output = weights * self.alpha
        return output
        
    def norm(self, weigths: np.array) -> float:
        """calculates the norm enhanced loss

        Args:
            weigths (np.array): weights

        Returns:
            float: loss
        """
        weigths_abs = np.multiply(weigths, weigths)
        output = np.sum(weigths_abs) * self.alpha
        return output