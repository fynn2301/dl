from cmath import sqrt
import copy
from doctest import OutputChecker
from re import X
import numpy as np
import matplotlib.pyplot as plt


class Checker():
    """Creates a Checkers board pattern
    """

    def __init__(self, resolution: int, tile_size: int) -> None:
        """Initilizes the checker pattern class

        Args:
            resolution (int): number of pixels in each dimension
            tile_size (int): number of pixels that build one tile
        """
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.ndarray((resolution, resolution))

        # avoiding truncated checkerboards
        if resolution % (tile_size*2) != 0:
            print(
                "WARNING: Checkersboard can't be build due to the odd relation of resolution and tile_size")

    def draw(self) -> np.ndarray:
        """Creates the checkers array

        Returns:
            np.ndarray: returns the array
        """
        # create one tile of each color
        tile_array_black = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
        tile_array_white = np.ones((self.tile_size, self.tile_size), dtype=np.uint8)
        #tile_array_white[:,:] = 255
        
        # create the upper and the lower row
        upper_row = np.concatenate((tile_array_black, tile_array_white), axis=1)
        lower_row = np.concatenate((tile_array_white, tile_array_black), axis=1)
        
        # create a 2x2 tile
        tile_array_2x2 = np.concatenate((upper_row, lower_row), axis=0)
        
        # create the complete checkersboard by repeating the 2x2
        repeat = self.resolution // (self.tile_size * 2)
        checkers_board = np.tile(tile_array_2x2, (repeat, repeat))
        self.output = checkers_board
        
        # makes a copy of the array and returns it
        return np.copy(checkers_board)

    def show(self) -> None:
        """Shows the pattern
        """
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()


class Circle():
    """Creates a Circle pattern
    """

    def __init__(self, resolution: int, radius: int, position:tuple) -> None:
        """Initilizes the class Checker
        """
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = np.ndarray((resolution, resolution))

    def draw(self) -> np.ndarray:
        """Creating the array with the circle

        Returns:
            np.ndarray: the created array
        """
        
        a = np.arange(self.resolution)  
        b = np.arange(self.resolution)  
        x, y = np.meshgrid(a, b, sparse=True)
        
        # x = array % self.resolution
        # y = array // self.resolution
        # sqrt(pow(x - x_0,2), pow(y - y_0,2)) < radius  -> the value is inside the radius and it should be white
        x_0 = self.position[0]
        y_0 = self.position[1]
        array = np.where(np.sqrt(np.power(x - x_0, 2) + np.power(y - y_0, 2)) <= self.radius, 1, 0)
        
        self.output = array
        return np.copy(array)

    def show(self) -> None:
        """Shows the pattern
        """
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()


class Spectrum():
    """Creates a Circle pattern
    """

    def __init__(self, resolution: int) -> None:
        """Initilizes the class Checker
        """
        self.resolution = resolution
        self.output = np.ndarray((resolution, resolution))

    def draw(self) -> np.ndarray:
        """Creating the array with the circle

        Returns:
            np.ndarray: the created array
        """
        spectrum = np.zeros((self.resolution, self.resolution, 3))
        spectrum[:, :, 0] = np.linspace(0.0, 1.0, self.resolution)
        spectrum[:, :, 1] = np.linspace(0.0, 1.0, self.resolution)[np.newaxis].T
        spectrum[:, :, 2] = np.linspace(1.0, 0.0, self.resolution)
        self.output = spectrum
        return np.copy(spectrum)

    def show(self) -> None:
        """Shows the pattern
        """
        plt.imshow(self.output)
        plt.axis('off')
        plt.show()
