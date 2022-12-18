import copy
import numpy as np
from Layers.Base import BaseLayer
import math

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape: tuple) -> None:
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
    
    def forward(self, input_tensor: np.array) -> np.array:
        """ Calculates the pooling of the input tensor

        Args:
            input_tensor (np.array): output of the last layer

        Returns:
            np.array: pooling of the input
        """
        self.input_shape = input_tensor.shape
        kernel_size_y= self.pooling_shape[0]
        kernel_size_x = self.pooling_shape[1]
        if len(self.stride_shape) == 1:
            stride_size_y = self.stride_shape[0]
            stride_size_x = self.stride_shape[0]
        else:
            stride_size_y = self.stride_shape[0]
            stride_size_x = self.stride_shape[1]
            
        batch_size, chanel_size, input_size_y, input_size_x = self.input_shape
        output_y_size = math.ceil((input_size_y - kernel_size_y + 1) / stride_size_y)
        output_x_size = math.ceil((input_size_x - kernel_size_x + 1) / stride_size_x)
        
        # init the output tensor
        output_tensor = np.zeros((batch_size, chanel_size, output_y_size, output_x_size))
        
        # for the backward methode we need to store the indices
        self.indices_tensor = np.zeros(output_tensor.shape + tuple([2]))
        
        # produce the indice maps for later sliceing of the input tensor
        index_list_upper_y = np.arange(kernel_size_y, input_size_y + 1, stride_size_y) - 1
        index_list_upper_y = np.tile(index_list_upper_y, (output_x_size, 1))
        index_list_upper_y = index_list_upper_y.T.flatten()
        
        index_list_upper_x = np.arange(kernel_size_x, input_size_x + 1, stride_size_x) - 1
        index_list_upper_x = np.tile(index_list_upper_x, (output_y_size, 1))
        index_list_upper_x = index_list_upper_x.flatten()
        
        c_index_list = np.zeros((output_y_size * output_x_size))
        c_index_list = c_index_list.astype(int)
        
        dy, dx = np.mgrid[-kernel_size_y+1:0:kernel_size_y * 1j, -kernel_size_x+1:0:kernel_size_x * 1j]
        dy = dy.astype(int)
        dx = dx.astype(int)
        
        # these arrays represent one index map in the input layer for every value in the output tensor in 2d these are flatten 1d arrays that will be reshaped in the end
        y_index_arrays = dy[None, :, :] + index_list_upper_y[:, None, None] 
        x_index_arrays = dx[None, :, :] + index_list_upper_x[:, None, None] 
        
        self.max_indices = np.zeros((batch_size, chanel_size, output_x_size * output_y_size, 2))
        
        for b, batch_tensor in enumerate(input_tensor):
            for c, chanel_tensor in enumerate(batch_tensor):
                # get the max value of each slice and store it in the output tensor
                input_sliced_array = chanel_tensor[y_index_arrays,x_index_arrays]
                max_array_flatten = np.amax(input_sliced_array, (1,2)).flatten()
                output_tensor[b,c,:,:] = max_array_flatten.reshape((output_y_size, output_x_size))
                
                # getting the index of every max value and store it in an array in the end
                max_index_vector = input_sliced_array.reshape(input_sliced_array.shape[0], -1).argmax(1)
                max_index_pair_vector = np.column_stack(np.unravel_index(max_index_vector, input_sliced_array[0,:,:].shape))
                
                # getting the ground variables of every kernel positioning
                self.max_indices[b,c,:,1] = np.arange(output_x_size * output_y_size)
                self.max_indices[b,c,:,1] = (self.max_indices[b,c,:,1] % output_x_size) * stride_size_x + (self.max_indices[b,c,:,1] // output_x_size) * input_size_x * stride_size_y
                
                self.max_indices[b,c,:,0] = np.arange(output_x_size * output_y_size)
                self.max_indices[b,c,:,0] = (self.max_indices[b,c,:,0] % output_x_size) * stride_size_x + (self.max_indices[b,c,:,0] // output_x_size) * input_size_x * stride_size_y
                
                self.max_indices[b,c,:,0] = self.max_indices[b,c,:,0] // input_size_x
                self.max_indices[b,c,:,1] = self.max_indices[b,c,:,1] % input_size_x
                
                # adding the position of the max values inside each kernel positioning to the global indices
                self.max_indices[b,c,:,1] += max_index_pair_vector[:,1]
                self.max_indices[b,c,:,0] += max_index_pair_vector[:,0]
                pass
                
        self.max_indices = self.max_indices.astype(int)
        return output_tensor
    
    def backward(self, error_tensor: np.array) -> np.array:
        """Propergating the error tensor back through the pooling layer

        Args:
            error_tensor (np.array): error tensor from the successor layer

        Returns:
            np.array: error tensor passed to the previous layer
        """
        error_tensor_0 = np.zeros(self.input_shape)
        
        # TODO: try to get rid of the loop
        # IDEA: Get all the max indices of all batches and chanels in one array and then 
        # add up all the indices that have the same index tuple and use:
        """
        batches, chanels, rows, cols = zip(*self.max_indices)
        """
        for b, batch_tensor in enumerate(error_tensor):
            for c, chanel_tensor in enumerate(batch_tensor):
                error_tensor_flatten = chanel_tensor.flatten()
                rows, cols = zip(*self.max_indices[b,c])
                
                error_tensor_0_chanel = error_tensor_0[b ,c]
                
                for row, col, val in zip(rows, cols, error_tensor_flatten): 
                    error_tensor_0_chanel[row, col] += val
                
        return error_tensor_0
        