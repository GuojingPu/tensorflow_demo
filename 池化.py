#!/usr/bin/python3
# -*- coding:utf-8 -*-

import tensorflow as tf
data  = tf.constant([[3.0,2.0,3.0,4.0], [2.0,6.0,2.0,4.0], [1.0,2.0,1.0,5.0], [4.0,3.0,2.0,1.0]])

print(data)

data = tf.reshape(data,[1,4,4,1])
"""
[
    [
        [[3.0], [2.0], [3.0], [4.0]], 
        [[2.0], [6.0], [2.0], [4.0]], 
        [[1.0], [2.0], [1.0], [5.0]], 
        [[4.0], [3.0], [2.0], [1.0]]
    ]
]
"""

print(data)

maxPolling = tf.nn.max_pool(value=data,
                            ksize=[1,2,2,1],  #池化窗口的大小[batch,wieght,height,channels]
                            strides=[1,2,2,1],#水平步长和竖直步长是2
                            padding='VALID')

with tf.Session() as sess:
    print(sess.run(maxPolling))
