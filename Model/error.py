import numpy as np


class error(object):

    def __init__(self, func, deriv):
        self.func = func
        self.deriv = deriv

    def forward(self, guess, data):
        return self.func(guess, data)

    def backward(self, guess, data):
        return self.deriv(guess, data)


class mse(error):

    def __init__(self):
        mse = lambda g, d: np.mean((g - d) ** 2)
        deriv = lambda g, d: (2 / np.size(d)) * (g - d)
        super().__init__(mse, deriv)


class mae(error):

    def __init__(self):
        mae = lambda g, d: np.mean(np.abs(g - d))

        def deriv(g, d):
            g[g > d] = 1
            g[g < d] = -1
            g[g == d] = 0
            return g

        super().__init__(mae, deriv)


class bce(error):

    def __init__(self):
        bce = lambda g, d: -np.mean(d * np.log(g) + (1 - d) * np.log(1 - g))
        deriv = lambda g, d: ((1 - d) / (1 - g) - d / g) / np.size(d)
        super().__init__(bce, deriv)