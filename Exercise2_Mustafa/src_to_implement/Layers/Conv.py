import copy
import numpy as np
from scipy.ndimage import correlate, correlate1d, convolve, convolve1d
from Layers.Base import BaseLayer
from Layers.Initializers import UniformRandom


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        self.kernel_width = 1
        if len(convolution_shape) == 3:
            self.kernel_width = convolution_shape[2]

        self.uniform_rand = UniformRandom()

        self.weights = self.uniform_rand.initialize(tuple([num_kernels]) + convolution_shape, convolution_shape[0] * convolution_shape[1] * self.kernel_width, num_kernels * convolution_shape[1] * self.kernel_width)
        self.bias = self.uniform_rand.initialize((num_kernels, 1), num_kernels, 1)

        self._optimizer_weights = None
        self._optimizer_bias = None

        self._gradient_weights = None
        self._gradient_bias = None

        self.input_tensor = None
        self.output_tensor = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @gradient_weights.deleter
    def gradient_weights(self):
        del self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @gradient_bias.deleter
    def gradient_bias(self):
        del self._gradient_bias

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

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = None

        if len(input_tensor.shape) == 3:
            # Get input shape
            batch_size, _, input_y = input_tensor.shape

            # Calculate output shape
            output_y = int(np.ceil(input_y / self.stride_shape[0]))

            # Initialize the output tensor
            output_tensor = np.zeros((batch_size, self.num_kernels, input_y))

            # Add bias to output tensor
            for batch_num in range(batch_size):
                for layer_num in range(self.num_kernels):
                    output_tensor[batch_num, layer_num] += self.bias[layer_num]

            # Loop over the input batches and correlate (without considering the stride shape)
            for batch_num, batch in enumerate(input_tensor):
                for channel_num, channel in enumerate(batch):
                    for kernel_num in range(self.num_kernels):
                        output_tensor[batch_num, kernel_num] += correlate1d(channel, self.weights[kernel_num, channel_num], mode='constant', cval=0.0)

            # Generate stride boolean array
            row, col = np.indices((1, input_y))
            stride_bool_array = np.where(col % self.stride_shape[0] == 0, True, False)
            # Fit the output_tensor
            output_tensor = np.reshape(output_tensor[:, :, stride_bool_array[0]], (batch_size, self.num_kernels, output_y))
        else:
            # Get input shape
            batch_size, _, input_y, input_x = input_tensor.shape

            # Calculate output shape
            output_y = int(np.ceil(input_y / self.stride_shape[0]))
            output_x = int(np.ceil(input_x / self.stride_shape[1]))

            # Initialize the output tensor
            output_tensor = np.zeros((batch_size, self.num_kernels, input_y, input_x))

            # Add bias to output tensor
            for batch_num in range(batch_size):
                for layer_num in range(self.num_kernels):
                    output_tensor[batch_num, layer_num] += self.bias[layer_num]

            # Loop over the input batches and correlate (without considering the stride shape)
            for batch_num, batch in enumerate(input_tensor):
                for channel_num, channel in enumerate(batch):
                    for kernel_num in range(self.num_kernels):
                        output_tensor[batch_num, kernel_num] += correlate(channel, self.weights[kernel_num, channel_num], mode='constant', cval=0.0)

            # Generate stride boolean array
            row, col = np.indices((input_y, input_x))
            stride_bool_array = np.where(row % self.stride_shape[0] == 0, True, False) & np.where(col % self.stride_shape[1] == 0, True, False)

            # Fit the output_tensor
            output_tensor = np.reshape(output_tensor[:, :, stride_bool_array], (batch_size, self.num_kernels, output_y, output_x))

        self.output_tensor = output_tensor
        return output_tensor

    def backward(self, error_tensor):
        if len(self.convolution_shape) == 2:
            # Calculate error_tensor for next layer
            # Rearrange the convolution kernels
            channel_size, m = self.convolution_shape
            _, channel_error, y_error = error_tensor.shape
            batch_size, _, y = self.input_tensor.shape

            new_kernels = np.zeros((channel_size, self.num_kernels, m))

            for batch_num in range(error_tensor.shape[0]):
                for channel_num in range(channel_size):
                    new_kernels[channel_num] = self.weights[:, channel_num]

            # Up-sample the error tensor
            upsample_error_tensor = np.zeros((batch_size, channel_error, y))
            for batch_num, batch in enumerate(error_tensor):
                for channel_num, channel in enumerate(batch):
                    for y_num, yy in enumerate(channel):
                        upsample_error_tensor[batch_num, channel_num, y_num * self.stride_shape[0]] = yy

            # Convolve over the output tensor from the previous forward
            error_tensor_prev = np.zeros((batch_size, channel_size, y))

            for batch_num, batch in enumerate(upsample_error_tensor):
                for channel_num, channel in enumerate(batch):
                    for kernel_num in range(new_kernels.shape[0]):
                        error_tensor_prev[batch_num, kernel_num] += convolve1d(channel,
                                                                              new_kernels[kernel_num, channel_num], mode='constant', cval=0.0)

            # Calculate the bias gradient
            self.gradient_bias = [np.sum(batch_error_tensor) for batch_error_tensor in error_tensor]

            # Calculate the weights gradient

            # Optimize the weights and the bias, if optimizers are initialized
            if self._optimizer_weights is not None and self._optimizer_bias is not None:
                # Update bias
                self.bias = self._optimizer_bias.calculate_update(self.bias, self.gradient_bias)

                # Update weights
                self.weights = self._optimizer_weights.calculate_update(self.weights, self.gradient_weights)
        else:
            # Calculate error_tensor for next layer
            # Rearrange the convolution kernels
            channel_size, m, n = self.convolution_shape
            _, channel_error, y_error, x_error = error_tensor.shape
            batch_size, _, y, x = self.input_tensor.shape

            new_kernels = np.zeros((channel_size, self.num_kernels, m, n))

            for batch_num in range(error_tensor.shape[0]):
                for channel_num in range(channel_size):
                    new_kernels[channel_num] = self.weights[:, channel_num]

            # Up-sample the error tensor
            upsample_error_tensor = np.zeros((batch_size, channel_error, y, x))
            for batch_num, batch in enumerate(error_tensor):
                for channel_num, channel in enumerate(batch):
                    for y_num, yy in enumerate(channel):
                        for x_num, xx in enumerate(yy):
                            upsample_error_tensor[batch_num, channel_num, y_num*self.stride_shape[0], x_num*self.stride_shape[1]] = xx

            # Convolve over the output tensor from the previous forward
            error_tensor_prev = np.zeros((batch_size, channel_size, y, x))

            for batch_num, batch in enumerate(upsample_error_tensor):
                for channel_num, channel in enumerate(batch):
                    for kernel_num in range(new_kernels.shape[0]):
                        error_tensor_prev[batch_num, kernel_num] += convolve(channel, new_kernels[kernel_num, channel_num], mode='constant', cval=0.0)

            # Calculate the bias gradient
            self.gradient_bias = [np.sum(b_error_tensor) for b_error_tensor in error_tensor]

            # TODO Calculate the weights gradient (self.gradient_weights)
            # TODO Loop over batch and channels
            # TODO Correlate over input_tensor with error_tensor as kernel (/weights)
            # TODO Cut the resulting output into the shape of the kernel (self.convolution_shape[1,2])

            # Optimize the weights and the bias, if optimizers are initialized
            if self._optimizer_weights is not None and self._optimizer_bias is not None:
                # Update bias
                self.bias = self._optimizer_bias.calculate_update(self.bias, self.gradient_bias)

                # Update weights
                self.weights = self._optimizer_weights.calculate_update(self.weights, self.gradient_weights)

        return error_tensor_prev

