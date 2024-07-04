import random

from Layers import fullyconnected as fc, convolution as cn, activation as ac, GradDescent as gd
from Model import error as er
import numpy as np


class pointer():

    def __init__(self, layer):
        self.current = layer
        self.next = None
        self.prev = None

    def getForward(self, input):
        return self.current.forward(input)

    def getBackward(self, grad, lr):
        return self.current.backward(grad, lr)


class queue():

    def __init__(self):
        self.first = None

    def enqueue(self, val):
        point = pointer(val)

        if self.first == None:
            self.first = point

        elif self.first.next == None:
            self.first.next = point
            self.value = point
            self.value.prev = self.first

        else:
            self.value.next = point
            previous_value = self.value
            self.value = point
            self.value.prev = previous_value

    def full_forward(self, input, train=True):
        self.curr_lay = self.first
        while True:
            if (not train and not isinstance(self.curr_lay.current, fc.dropout)) or (train):
                input = self.curr_lay.getForward(input)

            if self.curr_lay.next is not None:
                self.curr_lay = self.curr_lay.next
            else:
                break

        return input

    def full_backward(self, grad, lr):
        while True:
            grad = self.curr_lay.getBackward(grad, lr)

            if self.curr_lay.prev is not None:
                self.curr_lay = self.curr_lay.prev
            else:
                break


class learning_rate():

    def __init__(self, rate, update_func):
        self.learning_rate = rate
        self.update_func = update_func

    def update(self, epoch):
        self.learning_rate = self.update_func(lr = self.learning_rate, epoch = epoch)


class sequence():

    def __init__(self, orig_shape, descent):
        self.input_shape = orig_shape[1:]
        self.num_of_in = orig_shape[0]
        self.seq = queue()
        self.descent = descent

    def dense(self, outputs, activ, drop=0):
        self.seq.enqueue(fc.dense(self.input_shape[0], outputs, self.descent, activ))
        self.input_shape = (outputs, self.num_of_in)

        if drop != 0:
            self.seq.enqueue(fc.dropout(drop, self.input_shape))

        self.seq.enqueue(activ)

    def convolution(self, k_size, k_depth, activ, strides=0):
        conv_shape = lambda x: x - k_size + 1
        self.seq.enqueue(cn.convolution(k_size, k_depth, self.input_shape, self.descent, activ))
        self.seq.enqueue(activ)
        self.input_shape = (k_depth, conv_shape(self.input_shape[1]), conv_shape(self.input_shape[2]))

        if strides != 0:
            pool_shape = lambda x: int(np.ceil(x / strides))
            self.seq.enqueue(cn.max_pooling(strides, self.input_shape))
            self.input_shape = (k_depth, pool_shape(self.input_shape[1]), pool_shape(self.input_shape[2]))

    def reshape(self, output_shape):
        self.seq.enqueue(cn.reshape(self.input_shape, output_shape))
        output_shape = [output_shape]
        self.input_shape = (*output_shape, self.num_of_in)

    def flatten(self):
        self.seq.enqueue(cn.reshape(self.input_shape, (np.prod(self.input_shape), 1)))
        self.input_shape = (np.prod(self.input_shape), 1, self.num_of_in)

    def train(self, input, output, epoch, lr, error=er.mse):
        for e in range(epoch):
            guess = self.seq.full_forward(input)
            print("epoch:", e + 1)
            print("error: {0:.4f}".format(error.forward(guess, output)))
            print(*np.round(guess, 4)[0], "\n")

            self.seq.full_backward(error.backward(guess, output), lr.learning_rate)
            lr.update(e)

        guess = self.seq.full_forward(input, train = False)
        print("error: {0:.4f}".format(error.forward(guess, output)))
        print(*np.round(guess, 4)[0], "\n")

    def inference(self, input):
        guess = self.seq.full_forward(input, train = False)
        print(*np.round(guess, 4)[0], "\n")
