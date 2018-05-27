import math
import numpy as np
from PIL import Image

def basic_sigmoid(x):
    s = 1/(1+math.exp(-x))
    return s
print(1, basic_sigmoid(10))

x = np.array([1,2,3])
print(2, x+3)

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s
print(3, sigmoid(x))

def sigmoid_derivative(x):
    s = 1/(1+np.exp(-x))
    ds = s*(1-s)
    return ds
print(4, sigmoid_derivative(x))

def image2vector(image):
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2]), 1)
    return v
image = np.array(Image.open("./icon_sample.JPG"))
print(5, image2vector(image))

def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x/x_norm
    return x
x = np.array([
    [0,3,4], [1,6,4]
])
print(6, normalizeRows(x))

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp/x_sum
    return s
x = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
print(7, softmax(x))