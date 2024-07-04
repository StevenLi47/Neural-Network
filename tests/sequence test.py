from Model import sequence as sq, error as er
from Layers import activation as ac, GradDescent as gd
import cv2
import numpy as np


def update(lr, epoch):
    if epoch % 10 == 0:
        lr = 0.9 * lr

    return lr


image_reshape = lambda im: np.array(cv2.split(im))

dog1 = cv2.imread("dog1.jpg") / 255
dog2 = cv2.imread("dog2.jpg") / 255
dog3 = cv2.imread("dog3.jpg") / 255
dog4 = cv2.imread("dog4.jpg") / 255
dog5 = cv2.imread("dog5.jpg") / 255
dog6 = cv2.imread("dog7.jpg") / 255
cat1 = cv2.imread("cat1.jpg") / 255
cat2 = cv2.imread("cat2.jpg") / 255
cat3 = cv2.imread("cat3.jpg") / 255
cat4 = cv2.imread("cat4.jpg") / 255
cat5 = cv2.imread("cat5.jpg") / 255
cat6 = cv2.imread("cat6.jpg") / 255

images = [dog1, dog2, dog3, dog4, dog5, cat1, cat2, cat3, cat4, cat5]
infer_images = [dog6, cat6]

for i in range(len(images)):
    images[i] = image_reshape(images[i])

for i in range(len(infer_images)):
    infer_images[i] = image_reshape(infer_images[i])

outputs = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])


seq = sq.sequence(np.shape(images), gd.adam(0.9, 0.9))
seq.convolution(3, 4, activ = ac.relu())
seq.convolution(3, 4, strides = 2, activ = ac.relu())
seq.convolution(3, 8, activ = ac.relu())
seq.convolution(3, 8, strides = 2, activ = ac.relu())
seq.convolution(3, 16, activ = ac.relu())
seq.convolution(3, 16, strides = 2, activ = ac.relu())
seq.flatten()
seq.dense(100, drop = 0.7, activ = ac.relu())
seq.dense(100, drop = 0.7, activ = ac.relu())
seq.dense(1, activ = ac.sigmoid())

seq.train(images, outputs, 30, sq.learning_rate(0.0001, update), er.bce())
seq.inference(infer_images)
