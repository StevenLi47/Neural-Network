import numpy as np

class learning_rate():

    def __init__(self, rate, update_func):
        self.learning_rate = rate
        self.update_func = update_func

    def backwards(self, *args):
        self.learning_rate = self.update_func(lr = self.learning_rate, *args)