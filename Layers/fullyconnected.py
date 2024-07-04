import copy

import numpy as np
from Layers import activation as ac, GradDescent as gd


class dense():

    def __init__(self, inputs, outputs, descent, activ = None):
        if isinstance(activ, ac.relu):
            scale = np.sqrt(2 / inputs)
        elif isinstance(activ, ac.tanh) or isinstance(activ, ac.sigmoid):
            scale = np.sqrt(2 / (inputs + outputs))
        else:
            scale = 0.1

        self.weight = np.random.normal(loc=0, scale=scale, size=(outputs, inputs))
        self.bias = np.random.normal(loc=0, scale=scale, size=(outputs, 1))
        self.W_descent = copy.deepcopy(descent)
        self.b_descent = copy.deepcopy(descent)

    def forward(self, input):
        self.input = input
        return np.matmul(self.weight, self.input) + self.bias

    def backward(self, grad, lr):
        bias_grad = np.mean(grad, axis=1).reshape(-1, 1)
        self.bias -= lr * self.W_descent.backwards(bias_grad)
        weight_grad = np.matmul(grad, self.input.T)
        self.weight -= lr * self.b_descent.backwards(weight_grad)
        return np.matmul(self.weight.T, grad)


class dropout():

    def __init__(self, rate, shape):
        self.rate = rate
        self.shape = shape

    def forward(self, input):
        self.drop_out = np.random.choice([0, 1], self.shape, p = [1 - self.rate, self.rate]).astype(float)
        output = np.multiply(self.drop_out, input)
        return output

    def backward(self, grad, lr):
        return np.multiply(self.drop_out, grad) / self.rate