from Layers import fullyconnected as fc, convolution as cn, activation as ac, GradDescent as gd
from Model import error as er
import numpy as np


class pointer():

    def __init__(self, layer):
        self.current = layer
        self.next = None
        self.prev = None

    def setNext(self, layer):
        self.next = layer
        self.next.prev = self.current

    def getForward(self, input):
        return self.activ.forward((self.current.forward(input)))

    def getBackward(self, grad, lr):
        return self.current.backward(grad, lr)