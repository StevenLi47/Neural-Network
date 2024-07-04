from Layers import fullyconnected as fc, convolution as cn
from Model import error as er
import numpy as np


class pointer(object):

    def __init__(self):
        self.current = None
        self.next = []
        self.prev = []

    def setNext(self, *input):
        self.next.append(*input)
        self.next.prev = self.current

    def getForward(self, input):
        return self.current.forward(input)

    def getBackward(self, grad, lr):
        return self.current.backward(grad, lr)


class layer(pointer):

    def __init__(self, layer):
        super().__init__()
        self.current = layer



class queue(object):

    def __init__(self):
        self.cur = None
        self.first = None
        self.next = []
        self.prev = []

    def add(self, layer):
        if self.cur is None:
            self.first = pointer(layer)
            self.cur = self.first
        else:
            self.cur.setNext(pointer(layer))
            self.cur = self.cur.next

    def setNext(self, *seq):
        self.next.append(*seq)

    def full_forward(self, input, train=True):
        self.cur = self.first

        while self.cur is not None:
            if (not train and not isinstance(self.cur.current, fc.dropout)) or (train):
                input = self.cur.getForward(input)

            self.cur = self.cur.next

        return input

    def full_backward(self, grad, lr):
        while self.cur is not None:
            grad = self.cur.getBackward(grad, lr)
            self.cur = self.cur.prev


class sequence(queue):

    def __init__(self, in_shape, descent):
        super().__init__()
        self.input_shape = in_shape[1:]
        self.num_of_in = in_shape[0]
        self.descent = descent

    def dense(self, outputs, activ, drop = 0):
        self.add(fc.dense(self.input_shape[0], outputs, self.descent, activ))
        self.input_shape = (outputs, self.num_of_in)

        if drop != 0:
            self.add(fc.dropout(drop, self.input_shape))

        self.add(activ)

    def convolution(self, k_size, k_depth, activ, strides = 0):
        conv_shape = lambda x: x - k_size + 1
        self.add(cn.convolution(k_size, k_depth, self.input_shape, self.descent, activ))
        self.add(activ)
        self.input_shape = (k_depth, conv_shape(self.input_shape[1]), conv_shape(self.input_shape[2]))

        if strides != 0:
            pool_shape = lambda x: int(np.ceil(x / strides))
            self.add(cn.max_pooling(strides, self.input_shape))
            self.input_shape = (k_depth, pool_shape(self.input_shape[1]), pool_shape(self.input_shape[2]))

    def reshape(self, output_shape):
        self.add(cn.reshape(self.input_shape, output_shape))
        self.input_shape = (*[output_shape], self.num_of_in)

    def flatten(self):
        self.add(cn.reshape(self.input_shape, (np.prod(self.input_shape), 1)))
        self.input_shape = (np.prod(self.input_shape), 1, self.num_of_in)

    def train(self, input, output, epoch, lr, error=er.mse):
        for e in range(epoch):
            guess = self.full_forward(input)
            print("epoch:", e + 1)
            print("error: {0:.4f}".format(error.forward(guess, output)))
            print(*np.round(guess, 4)[0], "\n")

            self.full_backward(error.backward(guess, output), lr.learning_rate)
            lr.update(e)

        guess = self.full_forward(input, train = False)
        print("error: {0:.4f}".format(error.forward(guess, output)))
        print(*np.round(guess, 4)[0], "\n")

    def inference(self, input):
        guess = self.full_forward(input, train = False)
        print(*np.round(guess, 4)[0], "\n")


class learning_rate():

    def __init__(self, rate, update_func = lambda lr, epoch: lr):
        self.learning_rate = rate
        self.update_func = update_func

    def update(self, epoch):
        self.learning_rate = self.update_func(lr = self.learning_rate, epoch = epoch)
