import copy
import numpy as np
from scipy import signal
from Layers import activation as ac, GradDescent as gd

class convolution():

    def __init__(self, k_size, k_depth, in_shape, descent, activ = None):
        self.in_depth, self.in_length, self.in_width = in_shape
        self.k_size = k_size
        self.k_depth = k_depth

        if isinstance(activ, ac.relu):
            scale = np.sqrt(2 / (self.in_depth * self.k_size ** 2))
        elif isinstance(activ, ac.tanh) or np.isinstance(activ, ac.sigmoid):
            scale = np.sqrt(2 / ((self.in_depth + self.k_depth) * self.k_size ** 2))
        else:
            scale = 0.1

        self.kernels = np.random.normal(loc = 0, scale = scale, size = (self.k_depth, self.in_depth, self.k_size, self.k_size))
        self.bias = np.random.normal(loc = 0, scale = scale, size = (self.k_depth, self.in_length - self.k_size + 1, self.in_width - self.k_size + 1))
        self.W_descent = copy.deepcopy(descent)
        self.b_descent = copy.deepcopy(descent)

    def forward(self, input):
        self.input = input
        self.output = np.zeros((np.shape(input)[0], *np.shape(self.bias))) + np.copy(self.bias)

        for k in range(len(input)):
            for i in range(self.k_depth):
                for j in range(self.in_depth):
                    self.output[k, i] += signal.correlate2d(self.input[k][j], self.kernels[i, j], "valid")

        return self.output

    def backward(self, grad, lr):
        self.in_grad = np.zeros_like(self.input)
        kernel_grad = np.zeros((np.shape(self.input)[0], *np.shape(self.kernels)))

        for k in range(len(grad)):
            for i in range(self.k_depth):
                for j in range(self.in_depth):
                    kernel_grad[k, i, j] += signal.correlate2d(self.input[k][j], grad[k, i], "valid")
                    self.in_grad[k, j] += signal.convolve2d(grad[k, i], self.kernels[i, j], "full")

        self.kernels -= np.sum(lr * self.W_descent.backwards(kernel_grad), 0)
        self.bias -= np.sum(lr * self.b_descent.backwards(grad), 0)

        return self.in_grad


class max_pooling():

    def __init__(self, strides, in_shape):
        input_extend = lambda x: int(np.ceil(x / strides) * strides)
        self.strides = strides
        self.depth, self.length, self.width = in_shape
        self.extended_length, self.extended_width = input_extend(self.length), input_extend(self.width)
        self.pool_length, self.pool_width = int(self.extended_length / self.strides), int(self.extended_width / self.strides)

    def forward(self, input):
        self.max_pool = []
        self.input_split = []
        pool_shape = (self.depth, self.pool_length, self.pool_width)

        for i in input:
            extended_input = np.empty((self.depth, self.extended_length, self.extended_width))
            extended_input[:] = np.nan
            extended_input[:, :self.length, :self.width] = i

            for j in range(1, 4):
                extended_input = np.array(np.array_split(extended_input, pool_shape[-j], axis=2))

            self.input_split.append(extended_input)
            self.max_pool.append(np.nanmax(np.squeeze(extended_input, axis = 3), axis = (3, 4)))

        return self.max_pool

    def backward(self, grad_output, lr):
        grad = []
        reshape_dim = (int(np.size(self.input_split[0]) / self.strides ** 2), int(self.strides ** 2))

        for i in range(len(grad_output)):
            split_reshape = np.reshape(self.input_split[i], reshape_dim)
            pool_reshape = np.reshape(self.max_pool[i], (reshape_dim[0], 1))
            index_ar = np.floor(np.divide(split_reshape, pool_reshape, out = np.float32(np.ones_like(split_reshape)), where = pool_reshape != 0))
            gradient = np.reshape(np.multiply(index_ar, np.reshape(grad_output[i], (reshape_dim[0], 1))), np.shape(self.input_split[i]))


            for i in range(3):
                gradient = np.dstack(gradient)

            grad.append(gradient[:, :self.length, :self.width])

        return grad


class reshape():

    def __init__(self, input_shape, output_shape):
        self.in_shape = input_shape
        self.out_shape = output_shape

    def forward(self, input):
        input_reshape = []
        for i in input:
            input_reshape.append(np.reshape(i, self.out_shape))

        return np.hstack(input_reshape)

    def backward(self, output_grad, lr):
        original_shape = []
        for i in range(np.shape(output_grad)[-1]):
            original_shape.append(np.reshape(output_grad[:, i], self.in_shape))

        return original_shape