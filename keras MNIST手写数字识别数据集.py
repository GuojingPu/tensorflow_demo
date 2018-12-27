#!/usr/bin/python3
# -*- coding:utf-8 -*-
import  numpy as np
from keras.utils import np_utils
from keras.datasets import mnist
import pandas  as pd
import matplotlib
matplotlib.use('TkAgg')
import  matplotlib.pyplot as plt
#
#
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image,cmap='binary')
    plt.show()




def main():

    np.random.seed(10)

    (x_train_image, y_train_lable), (x_test_image, y_test_lable) = mnist.load_data()

    print(x_train_image.shape, y_train_lable.shape)
    print(x_test_image.shape, y_test_lable.shape)

    plot_image(x_train_image[0])



if __name__ == '__main__':
    main()





np.random.seed(10)

(x_train_image, y_train_lable), (x_test_image, y_test_lable) = mnist.load_data()

print(x_train_image.shape, y_train_lable.shape)
print(x_test_image.shape, y_test_lable.shape)