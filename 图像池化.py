#!/usr/bin/python3
# -*- coding:utf-8 -*-
import tensorflow as tf
import cv2
import numpy as np

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


img = cv2.imread('123.jpg')
img = np.array(img,dtype=np.float32)
x_image = tf.reshape(img,[1,512,512,3])#1张图片 图片大小512X512  图片通道3个

filter = tf.Variable(tf.ones([7,7,3,1]))#卷积核大小 7X7 图片3通道，1个卷积核


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    res = tf.nn.conv2d(x_image,filter,strides=[1,2,2,1],padding="SAME")#水平步长2 竖直步长2
    res = tf.nn.max_pool(res,[1,2,2,1],[1,2,2,1],padding='VALID')
    res_image = sess.run(tf.reshape(res,[128,128]))/128 +1
    print(res_image.shape)


cv2.imshow("test",res_image.astype('uint8'))
# plot_image(res_image.astype('uint8'))
cv2.waitKey()
