import numpy as np
import scipy.io
import scipy.misc

test_y = [1, 2, 3, 4, 5, 6]


for (test, rg) in zip(test_y, range(1, 3)):
    print(test)
    print(rg)


content_image = scipy.misc.imread("images/louvre_small.jpg")
print(content_image.shape)
print((1,) + content_image.shape)       # 添加一个维度
print(content_image)

MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
print(MEANS)

image = np.reshape(content_image, ((1,) + content_image.shape))
image = image - MEANS
print(image)