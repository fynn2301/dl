import numpy as np
class Constant():
    def __init__(self, weight_value:float = 0.1) -> None:
        self.weight_init_value = weight_value
    
    def initialize(self, weights_shape: tuple, fan_in: int, fan_out: int) -> np.array:
        """create a initialized weight tensor

        Args:
            weights_shape (tuple): _description_
            fan_in (int): _description_
            fan_out (int): _description_

        Returns:
            np.array: _description_
        """
        return np.full((fan_in, fan_out), self.weight_init_value)
    
class UniformRandom():
    def __init__(self) -> None:
        pass
    
    def initialize(self, weights_shape: tuple, fan_in: int, fan_out: int) -> np.array:
        """create a initialized weight tensor

        Args:
            weights_shape (tuple): _description_
            fan_in (int): _description_
            fan_out (int): _description_

        Returns:
            np.array: _description_
        """
        return np.random.rand(*weights_shape)
    
class Xavier():
    def __init__(self) -> None:
        pass
    
    def initialize(self, weights_shape: tuple, fan_in: int, fan_out: int) -> np.array:
        """create a initialized weight tensor

        Args:
            weights_shape (tuple): _description_
            fan_in (int): _description_
            fan_out (int): _description_

        Returns:
            np.array: _description_
        """
        sigma = np.sqrt(2 / (fan_in + fan_out))
        weights = np.random.normal(scale=sigma, size=weights_shape)
        return weights
    
class He():
    def __init__(self) -> None:
        pass
    
    def initialize(self, weights_shape: tuple, fan_in: int, fan_out: int) -> np.array:
        """create a initialized weight tensor

        Args:
            weights_shape (tuple): _description_
            fan_in (int): _description_
            fan_out (int): _description_

        Returns:
            np.array: _description_
        """
        sigma = np.sqrt(2 / fan_in)
        weights = np.random.normal(scale=sigma, size=weights_shape)
        return weights