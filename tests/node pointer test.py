import numpy as np
from copy import copy

class node():

    def __init__(self, input):
        self.input = input
        self.in_shape = np.shape(input)
        self.next = None
        self.prev = None

    def forward(self, input):
        return input

    def backward(self, grad, lr):
        return grad

    def setNext(self, next_node = None):
        self.next = next_node

        if next_node is not None:
            next_node.prev = self

    def setPrev(self, prev_node = None):
        self.prev = prev_node

        if prev_node is not None:
            prev_node.next = self

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.setNext()
        result.setPrev()
        return result


class sequence():

    def __init__(self):
        self.buffer = node(None)
        self.current = self.buffer
        self.length = 0
        self.position = -1
        self.forward = True

    def __iter__(self):
        return self

    def __next__(self):
        if self.position == -1:
            self.current = self.buffer

        if self.forward:
            self.current = self.current.next
            self.position += 1
        else:
            self.current = self.current.prev
            self.position += 1

        if self.position != len(self):
            return self.current
        else:
            self.position = -1
            raise StopIteration

    def __getitem__(self, item):
        self.current = self.buffer
        self.position = -1

        if isinstance(item, int):
            for i in range(item + 1):
                next(self)

            self.position = -1
            return self.current

        elif isinstance(item, slice):
            start, end, step = item.indices(len(self))
            sub_seq = sequence()

            for i in range(start, end, step):
                node_copy = copy(self[i])
                sub_seq.add(node_copy)

            return sub_seq

    def __len__(self):
        return self.length

    def __reversed__(self):
        seq_copy = copy(self)
        seq_copy.change_direct()
        return seq_copy

    def change_direct(self):
        if self.forward:
            self.forward = False
        else:
            self.forward = True

    def add(self, node):
        self.current.setNext(node)
        self.current = node
        self.current.setNext(self.buffer)
        self.length += 1

    def pop(self):
        self.buffer.setPrev(self.buffer.prev.prev)
        self.length -= 1

    def full_forward(self, input):
        for node in self:
            input = node.forward(input)

        return input

    def full_backward(self, grad, lr):
        self.change_direct()
        for node in self:
            grad = node.backward(grad, lr)
        self.change_direct()

    def train(self, input, output, epoch, lr, error):
        for e in range(epoch):
            guess = self.full_forward(input)
            print("epoch:", e + 1)
            print("error: {0:.4f}".format(error.forward(guess, output)))
            print(*np.round(guess, 4)[0], "\n")

            self.full_backward(error.backward(guess, output), lr.learning_rate)
            lr.update(e)

        guess = self.full_forward(input)
        print("error: {0:.4f}".format(error.forward(guess, output)))
        print(*np.round(guess, 4)[0], "\n")

    def inference(self, input):
        guess = self.full_forward(input)
        print(*np.round(guess, 4)[0], "\n")

seq = sequence()
seq.add(node(1))
seq.add(node(2))
seq.add(node(3))
seq.add(node(4))
seq.add(node(5))

seq2 = seq[0:4:2]
for i in seq2:
    print(i.input)

