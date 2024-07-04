import numpy as np


class activation(object):

    def __init__(self, func, deriv):
        self.func = func
        self.deriv = deriv

    def forward(self, input):
        self.input = input
        self.output = self.func(input)
        return self.output

    def backward(self, grad, lr):
        return np.multiply(grad, self.deriv(self.input))


class tanh(activation):

    def __init__(self):
        tanh = lambda x: np.tanh(x)
        deriv = lambda y: 1 - np.square(tanh(y))
        super().__init__(tanh, deriv)


class relu(activation):

    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        deriv = lambda y: np.clip(np.ceil(y), 0, 1)
        super().__init__(relu, deriv)


class sigmoid(activation):

    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-np.array(x)))
        deriv = lambda y: sigmoid(y) * (1 - sigmoid(y))
        super().__init__(sigmoid, deriv)
