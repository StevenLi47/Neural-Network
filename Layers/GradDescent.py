import numpy as np

class grad_descent():

    def backwards(self, grad):
        return grad


class momentum():

    def __init__(self, beta):
        self.beta = beta
        self.cache = None

    def backwards(self, grad):
        if self.cache is None:
            self.cache = np.zeros_like(grad)

        self.cache = self.beta * self.cache + (1 - self.beta) * grad
        return self.cache


class RMSprop():

    def __init__(self, beta, epsilon = 1e-8):
        self.beta = beta
        self.epsilon = epsilon
        self.cache = None

    def backwards(self, grad):
        if self.cache is None:
            self.cache = np.zeros_like(grad)

        self.cache = self.beta * self.cache + (1 - self.beta) * np.square(grad)
        return grad / (np.sqrt(self.cache) + self.epsilon)

class adam():

    def __init__(self, beta1, beta2, epsilon = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.v_cache = None
        self.p_cache = None
        self.timestep = 0

    def backwards(self, grad):
        if self.v_cache is None:
            self.v_cache = np.zeros_like(grad)
            self.p_cache = np.zeros_like(grad)

        self.timestep += 1

        self.v_cache = (self.beta1 * self.v_cache + (1 - self.beta1) * grad)
        self.v_cache_corrected = self.v_cache / (1 - self.beta1 ** self.timestep)

        self.p_cache = (self.beta2 * self.p_cache + (1 - self.beta2) * np.square(grad))
        self.p_cache_corrected = self.p_cache / (1 - self.beta2 ** self.timestep)

        return self.v_cache_corrected / (np.sqrt(self.p_cache_corrected + self.epsilon))

