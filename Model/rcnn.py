import numpy as np
from Model import sequence as sq
from Layers import activation as ac, convolution as cn, fullyconnected as fc, GradDescent as gd


class rpn():

    def __init__(self, k_size, feature_map):
        self.feature = feature_map
        self.depth, self.length, self.width = np.shape(feature_map)
        self.sliding_window = sq.pointer(cn.convolution(k_size, self.depth, (self.depth, self.length, self.width), gd.adam(0.9, 0.9), activ = ac.relu()))
        self.cls = cn.convolution(1, self.depth, anchor_depth, gd.adam(0.9, 0.9), activ = ac.relu())
        self.reg = cn.convolution(1, self.depth, anchor_depth, gd.adam(0.9, 0.9), activ = ac.relu())


class faster_rcnn():

    def __init__(self, feature_map, k_size, anchor_sizes):
        self.feature = feature_map
        self.depth, self.length, self.width = np.shape(feature_map)

        anchor_depth = len(anchor_sizes)
        self.anchor_map = np.zeros((anchor_depth, self.length, self.width, 2, 2))

        for layer in range(anchor_depth):
            for y in range(self.length):
                for x in range(self.width):
                    anchor_y, anchor_x = anchor_sizes[layer]
                    self.anchor_map[layer, y, x] = [[y, x], [y + anchor_y, x + anchor_x]]

        self.rpn_kernel = cn.convolution(k_size, self.depth, anchor_depth, gd.adam(0.9, 0.9), activ = ac.relu())
        self.cls_kernel = cn.convolution(1, self.depth, anchor_depth, gd.adam(0.9, 0.9), activ = ac.relu())
        self.reg_kernel = cn.convolution(1, self.depth, anchor_depth, gd.adam(0.9, 0.9), activ = ac.relu())

    def rpn(self, k_size = 3):
        scale = np.sqrt(2 / (self.depth * k_size ** 2))
        self.rpn_kernel = np.random.normal(loc = 0, scale = scale, size = (self.depth, self.depth, self.k_size, self.k_size))
        self.cls_kernel = np.random.normal(loc=0, scale = scale, size=(self.depth, self.depth, 1, 1))
        self.reg_kernel = np.random.normal(loc=0, scale = scale, size=(self.depth, self.depth, 1, 1))