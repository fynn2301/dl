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
        self.bias = np.reshape(self.bias, (self.num_kernels,))

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

    def initialize(self, weights_init, bias_init):
        self.weights = weights_init.initialize(tuple([self.num_kernels]) + self.convolution_shape, self.convolution_shape[0] * self.convolution_shape[1] * self.kernel_width, self.num_kernels * self.convolution_shape[1] * self.kernel_width)
        self.bias = bias_init.initialize((self.num_kernels, 1), self.num_kernels, 1)
        self.bias = np.reshape(self.bias, (self.num_kernels,))

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
            batch_size, channel_input, y = self.input_tensor.shape

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
                        error_tensor_prev[batch_num, kernel_num] += convolve1d(channel, new_kernels[kernel_num, channel_num], mode='constant', cval=0.0)

            # Calculate the bias gradient
            self.gradient_bias = np.sum(error_tensor, (0,2))
            #print(self.gradient_bias)
            #self.gradient_bias = np.mean(self.gradient_bias, (0))
            #print(self.gradient_bias)
            self.gradient_weights = np.zeros((self.num_kernels, *self.convolution_shape))

            # calculate the bias
            for error_batch, input_batch in zip(upsample_error_tensor, self.input_tensor):
                for kernel_num, error_kernel in enumerate(error_batch):
                    for chanel_num, input_chanel in enumerate(input_batch):
                        padded_gradient = correlate1d(input_chanel, error_kernel, mode="constant", cval=0)

                        gradient_y_size = self.convolution_shape[1]
                        input_y_size = input_chanel.shape[0]

                        cutoff_y_top = ((input_y_size - 1) // 2) - ((gradient_y_size - 1) // 2)
                        cutoff_y_bot = input_y_size - cutoff_y_top - gradient_y_size

                        gradient = padded_gradient[cutoff_y_top:-cutoff_y_bot]
                        self.gradient_weights[kernel_num, chanel_num, :] += gradient

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
            batch_size, channel_input, y, x = self.input_tensor.shape

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
            self.gradient_bias = np.sum(error_tensor, (0, 2, 3))
            #self.gradient_bias = np.mean(self.gradient_bias, (0))
            #self.gradient_bias = np.reshape(self.gradient_bias, (self.num_kernels, 1))
            self.gradient_weights = np.zeros((self.num_kernels, *self.convolution_shape))

            # calculate the bias
            for error_batch, input_batch in zip(upsample_error_tensor, self.input_tensor):
                for kernel_num, error_kernel in enumerate(error_batch):
                    for chanel_num, input_chanel in enumerate(input_batch):
                        padded_gradient = correlate(input_chanel, error_kernel, mode="constant", cval=0)

                        gradient_y_size = self.convolution_shape[1]
                        gradient_x_size = self.convolution_shape[2]
                        input_y_size = input_chanel.shape[0]
                        input_x_size = input_chanel.shape[1]

                        cutoff_y_top = ((input_y_size - 1) // 2) - ((gradient_y_size - 1) // 2)
                        cutoff_y_bot = input_y_size - cutoff_y_top - gradient_y_size

                        cutoff_x_left = ((input_x_size - 1) // 2) - ((gradient_x_size - 1) // 2)
                        cutoff_x_right = input_x_size - cutoff_x_left - gradient_x_size

                        gradient = padded_gradient[cutoff_y_top:-cutoff_y_bot, cutoff_x_left:-cutoff_x_right]
                        self.gradient_weights[kernel_num, chanel_num, :, :] += gradient

            # Optimize the weights and the bias, if optimizers are initialized
            if self._optimizer_weights is not None and self._optimizer_bias is not None:
                # Update bias
                self.bias = self._optimizer_bias.calculate_update(self.bias, self.gradient_bias)

                # Update weights
                self.weights = self._optimizer_weights.calculate_update(self.weights, self.gradient_weights)

        return error_tensor_prev

