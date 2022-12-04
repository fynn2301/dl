import numpy as np


class Constant:
    def __init__(self, constant_val=0.1):
        self.constant_val = constant_val

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full((fan_in, fan_out), self.constant_val)


class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.rand(*weights_shape)


class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_out + fan_in))
        return np.random.normal(0, sigma, size=weights_shape)


class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / fan_in)
        return np.random.normal(0, sigma, size=weights_shape)

