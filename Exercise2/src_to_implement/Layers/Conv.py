import copy
import math
import numpy as np
from Layers.Base import BaseLayer
from Layers.Initializers import UniformRandom

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels) -> None:
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        
        self._optimizer_weights = None
        self._optimizer_bias = None
        
        self._gradiant_weights = None
        self._gradiant_bias = None
        
        initializer = UniformRandom()
        self.weights = initializer.initialize(tuple([num_kernels]) + convolution_shape, convolution_shape[0], convolution_shape[1])
        self.bias = initializer.initialize((num_kernels,1),num_kernels, 1)

    @property
    def gradiant_weights(self):
        return self._gradiant_weights

    @gradiant_weights.setter
    def gradiant_weights(self, value):
        self._gradiant_weights = value

    @gradiant_weights.deleter
    def gradiant_weights(self):
        del self._gradiant_weights
        
        
    @property
    def gradiant_bias(self):
        return self._gradiant_bias

    @gradiant_bias.setter
    def gradiant_bias(self, value):
        self._gradiant_bias = value

    @gradiant_bias.deleter
    def gradiant_bias(self):
        del self._gradiant_bias
        
        
    @property
    def optimizer(self):
        return self._optimizer_bias

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer_weights = copy.deepcopy(value)
        self._optimizer_bias = copy.deepcopy(value)

    @optimizer.deleter
    def optimizer(self):
        del self._optimizer_bias
        del self._optimizer_weights
        
    def initialize(self, weights_init, biases_init) -> None:
        """Reinitialize the weights of the weights and biases

        Args:
            weights_init (init class): An instance of an initilizer class
            biases_init (init class): An instance of an initilizer class
        """
        #TODO:
        pass
        
    def forward(self, input_tensor: np.array) -> np.array:
        """Calculates the convolution and returns the input tensor for the next layer

        Args:
            input_tensor (np.array): input

        Returns:
            np.array: input fo r next layer
        """
        input_shape = input_tensor.shape
        if len(input_shape) == 3:
            # the input array is 1D
            return self._forward1d(input_tensor)
            
        else:
            # the input array is 2D
            return self._forward2d(input_tensor)
    
    def _forward1d(self, input_tensor: np.array) -> np.array:
        """This is a helper function that only works on 1d input convelution

        Args:
            input_tensor (np.array): this is a input array of size b x c x y

        Returns:
            np.array: output tensor
        """
        # get all different sizes for later uses
        batch_size, chanel_size, input_size_y = input_tensor.shape
        kernel_size_y= self.convolution_shape[1]
        
        stride_size_y = self.stride_shape[0]
        
        # get outputsize and add padding if needed in the right size
        output_y_size = 1 + (input_size_y - (kernel_size_y % 2)) // stride_size_y
        
        padding_y_total = (kernel_size_y + stride_size_y * (output_y_size - 1)) - input_size_y
        
        padding_y_top = math.ceil(padding_y_total / 2)
        padding_y_bot = padding_y_total - padding_y_top
        
        input_tensor = np.pad(input_tensor, [(0,0),(0,0),(padding_y_top, padding_y_bot)])
        
        # init the output tensor
        output_tensor = np.zeros((batch_size, self.num_kernels, output_y_size))
        _, _, input_size_y = input_tensor.shape
        
        # produce the indice maps for later sliceing of the input tensor
        index_list_upper_y = np.arange(kernel_size_y, input_size_y + 1, stride_size_y) - 1
        dc, dy = np.mgrid[0:chanel_size-1:chanel_size * 1j, -kernel_size_y+1:0:kernel_size_y * 1j]
        dc = dc.astype(int)
        dy = dy.astype(int)
        c_index_list = np.zeros((output_y_size))
        c_index_list = c_index_list.astype(int)
        
        # these arrays represent one index map in the input layer for every value in the output tensor
        chanel_index_arrays = dc[None, :, :] + c_index_list[:, None, None]
        y_index_arrays = dy[None, :, :] + index_list_upper_y[:, None, None] 
        
        # iterate of the kernels of each batch_tensor
        for b, batch_tensor in enumerate(input_tensor):
            for k, (weights, bias) in enumerate(zip(self.weights,self.bias)):
                # get the slices of the input tensor for every value in the output tensor
                input_sliced_array = batch_tensor[chanel_index_arrays,y_index_arrays]
                
                # multiply these sliced arrays with the weights and then add the bias of the kernel
                input_sliced_weights = np.multiply(input_sliced_array, weights)
                output_tensor_kernel = np.sum(input_sliced_weights, (2, 1)) + bias
                
                # write the solution to the output tensor
                output_tensor[b, k, :] = output_tensor_kernel
        return output_tensor
    
    def _forward2d(self, input_tensor: np.array) -> np.array:
        """This is a helper function that only works on 1d input convelution

        Args:
            input_tensor (np.array): this is a input array of size b x c x y

        Returns:
            np.array: output tensor
        """
        # get all different sizes for later uses
        batch_size, chanel_size, input_size_y, input_size_x = input_tensor.shape
        kernel_size_y= self.convolution_shape[1]
        kernel_size_x = self.convolution_shape[2]
        
        if len(self.stride_shape) == 1:
            stride_size_x = self.stride_shape[0]
            stride_size_y = self.stride_shape[0]
        else:
            stride_size_y, stride_size_x = self.stride_shape
        
        # get outputsize and add padding if needed in the right size
        output_y_size = 1 + (input_size_y - 1) // stride_size_y
        output_x_size = 1 + (input_size_x - 1) // stride_size_x
        
        padding_y_total = (kernel_size_y + stride_size_y * (output_y_size - 1)) - input_size_y
        padding_x_total = (kernel_size_x + stride_size_x * (output_x_size - 1)) - input_size_x
        
        padding_y_top = math.ceil(padding_y_total / 2)
        padding_y_bot = padding_y_total - padding_y_top
        
        padding_x_left = math.ceil(padding_x_total / 2)
        padding_x_right = padding_x_total - padding_x_left
        
        input_tensor = np.pad(input_tensor, [(0,0),(0,0),(padding_y_top, padding_y_bot), (padding_x_left, padding_x_right)])
        _, _, input_size_y, input_size_x = input_tensor.shape
        # init the output tensor
        output_tensor = np.zeros((batch_size, self.num_kernels, output_y_size, output_x_size))
        
        # produce the indices maps for later sliceing of the input tensor
        index_list_upper_y = np.arange(kernel_size_y, input_size_y + 1, stride_size_y) - 1
        index_list_upper_y = np.tile(index_list_upper_y, (output_x_size, 1))
        index_list_upper_y = index_list_upper_y.T.flatten()
        
        index_list_upper_x = np.arange(kernel_size_x, input_size_x + 1, stride_size_x) - 1
        index_list_upper_x = np.tile(index_list_upper_x, (output_y_size, 1))
        index_list_upper_x = index_list_upper_x.flatten()
        
        c_index_list = np.zeros((output_y_size * output_x_size))
        c_index_list = c_index_list.astype(int)
        
        dc, dy, dx = np.mgrid[0:chanel_size-1:chanel_size * 1j, -kernel_size_y+1:0:kernel_size_y * 1j, -kernel_size_x+1:0:kernel_size_x * 1j]
        
        dc = dc.astype(int)
        dy = dy.astype(int)
        dx = dx.astype(int)
        
        # these index maps contain one map for flatten for every cell in the output array
        chanel_index_arrays = dc[None, :, :, :] + c_index_list[:, None, None, None]
        y_index_arrays = dy[None, :, :, :] + index_list_upper_y[:, None, None, None] 
        x_index_arrays = dx[None, :, :, :] + index_list_upper_x[:, None, None, None] 
        
        # iterate of the chanels of each batch_tensor
        for b, batch_tensor in enumerate(input_tensor):
            for k, (weights, bias) in enumerate(zip(self.weights,self.bias)):
                # get the slices of the input tensor for every value in the output tensor
                input_sliced_array = batch_tensor[chanel_index_arrays,y_index_arrays,x_index_arrays]
                
                # multiply these sliced arrays with the weights and then add the bias of the kernel
                input_sliced_weights = np.multiply(input_sliced_array, weights)
                output_tensor_kernel = np.sum(input_sliced_weights, (3, 2, 1)) + bias
                
                # reshape the flatten arrays back to the wanted shape and write it to the output
                output_tensor_kernel = output_tensor_kernel.reshape(output_y_size,output_x_size)
                output_tensor[b, k, :, :] = output_tensor_kernel
        return output_tensor
    
    def backward(self, error_tensor: np.array) -> np.array:
        """Backpropagates through the convolution layer

        Args:
            error_tensor (np.array): error tensor

        Returns:
            np.array: error tensor for the previous layer
        """
        weights_T = self.weights.T
        error_tensor_0 = np.tensordot(weights_T, error_tensor, 2)
        pass