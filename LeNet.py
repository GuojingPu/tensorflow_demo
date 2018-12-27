#!/usr/bin/python3
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import time
x = tf.placeholder('float',[None,784])
y_ = tf.placeholder('float',[None,10])

x_image = tf.reshape(x,[-1,28,28,1])

print('x_image:',x_image)


"""1 卷积层1"""
#第一层卷积核
filter1 = tf.Variable(tf.truncated_normal([5,5,1,6]))
#第一层输出误差
bias1 = tf.Variable(tf.truncated_normal([6]))
print('**',filter1,bias1)

#第一次卷积运算
conv1 = tf.nn.conv2d(x_image,filter1,strides=[1,1,1,1],padding="SAME")
print('conv1:',conv1)

"""2 激活层1"""
h_conv1 = tf.nn.sigmoid(conv1 + bias1)
print('h_conv1:',h_conv1)

"""3 池化层2"""
maxPool2 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

print('maxPool2:',maxPool2)

"4 卷积层2"
filter2 = tf.Variable(tf.truncated_normal([5,5,6,16]))
#第一层输出误差
bias2 = tf.Variable(tf.truncated_normal([16]))
print('**',filter2,bias2)

#第二次卷积运算
conv2 = tf.nn.conv2d(maxPool2,filter2,strides=[1,1,1,1],padding="SAME")
print('conv2:',conv2)

"""5 激活层2 """
h_conv2 = tf.nn.sigmoid(conv2 + bias2)
print('h_conv2:',h_conv2)

"""6 池化层3"""
maxPool3 = tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
print('maxPool3:',maxPool3)


"7 卷积层3"
filter3 = tf.Variable(tf.truncated_normal([5,5,16,120]))
#第一层输出误差
bias3 = tf.Variable(tf.truncated_normal([120]))
print('**',filter3,bias3)
#卷积运算
conv3 = tf.nn.conv2d(maxPool3,filter3,strides=[1,1,1,1],padding="SAME")
print('conv2:',conv2)

"""8 激活层3"""
h_conv3 = tf.nn.sigmoid(conv3 + bias3)
print('h_conv3:',h_conv3)


"""全连接层"""
#权值参数
W_fc1 = tf.Variable(tf.truncated_normal([7*7*120,80]))

print('权值参数W_fc1:',W_fc1)
#偏置值
b_fc1= tf.Variable(tf.truncated_normal([80]))
print('偏置值b_fc1:',b_fc1)
#将卷积输出展开
h_pool2_flat = tf.reshape(h_conv3,[-1,7*7*120])

#神经网络运算 并添加sigmoid激活函数
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
print('h_fc1:',h_fc1)

"""输出层"""
#权值参数
W_fc2 = tf.Variable(tf.truncated_normal([80,10]))
#偏置值
b_fc2= tf.Variable(tf.truncated_normal([10]))

# y_conv = tf.maximum(tf.nn.softmax(tf.matmul(h_fc1,W_fc2) + b_fc2),le-30)
#使用sofrmax进行多分类
y_conv = tf.nn.softmax(tf.matmul(h_fc1,W_fc2) + b_fc2)
print('y_conv:',y_conv)

#损失函数熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

#使用梯度下降算法对模型进行训练
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)




sess = tf.InteractiveSession()

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))


sess.run(tf.global_variables_initializer())

mnist_data_set = input_data.read_data_sets("MNIST_data/",one_hot=True)
# mnist_data_set = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

start_time = time.time()

print('start_time:',start_time)
for i in range(20000):
    batch_xs,batch_ys = mnist_data_set.train.next_batch(200)

    if i % 2 == 0 :
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs,y_:batch_ys})
        print("step %d,training accuracy %g"%(i,train_accuracy))

        #计算时间间隔
        end_time = time.time()
        print('间隔time:',(end_time-start_time))
        start_time = end_time

    #训练数据
    train_step.run(feed_dict={x:batch_xs,y_:batch_ys})


sess.close()

