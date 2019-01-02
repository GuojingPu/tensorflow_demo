#!/usr/bin/python3
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import matplotlib
matplotlib.use('TkAgg')
import  matplotlib.pyplot as plt


"""
初始化卷积核变量的初始值。
从截断的正态分布中输出随机值。生成的值服从具有指定
平均值和标准偏差的正态分布，如果生成的值大于平均值
2个标准偏差的值则丢弃重新选择。这里正太分布的标准差为0.1。

"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    variable = tf.Variable(initial)
    return variable


"""
初始化单个卷积核上的偏置值
"""
def bias_varible(shape):
    initial = tf.constant(0.1,shape=shape) #所有的偏置值初始化为0.1
    varible=tf.Variable(initial)
    return varible


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")



def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')



def model_LeNet(x_image,y_):
    """
    LeNet模型
    :return: 
    """
    """ 卷积层1"""
    filter1 = weight_variable([5,5,1,32]) #第一层卷积核
    bias1 = bias_varible([32]) #第一层输出误差
    conv1 = conv2d(x_image,filter1) #第一次卷积运算
    h_conv1 = tf.nn.relu(conv1 + bias1)
    maxPool2 = max_pool_2x2(h_conv1)

    "卷积层2"
    filter2 = weight_variable([5,5,32,64])
    bias2 = bias_varible([64])
    conv2 = conv2d(maxPool2,filter2) #第二次卷积运算
    h_conv2 = tf.nn.relu(conv2 + bias2)
    maxPool3 = max_pool_2x2(h_conv2)

    # "卷积层3"
    # filter3 = weight_variable([5,5,16,120])
    # bias3 = bias_varible([120])#第三层输出误差/偏置值
    # conv3 = conv2d(maxPool3,filter3) #卷积运算
    # h_conv3 = tf.nn.relu(conv3 + bias3)

    """全连接层"""

    W_fc1 = weight_variable([7*7*64,1024]) #权值参数
    b_fc1= bias_varible([1024]) #偏置值
    h_pool2_flat = tf.reshape(maxPool3,[-1,7*7*64]) #将卷积输出展开
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1) #神经网络运算 并添加sigmoid激活函数


    W_fc2 = weight_variable([1024,128]) #权值参数
    b_fc2= bias_varible([128]) #偏置值
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc2)


    W_fc3 = weight_variable([128,10]) #权值参数
    b_fc3= bias_varible([10]) #偏置值

    """输出层"""
    y_conv = tf.nn.softmax(tf.matmul(h_fc2,W_fc3) + b_fc3); print('y_conv:',y_conv) #使用sofrmax进行多分类

    """损失函数"""

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))  #损失函数/熵

    """训练模型"""
    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy) #使用梯度下降算法对模型进行训练

    return y_conv,train_step



def accuracy_cal(y_,y_conv):

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy


def main():
    x = tf.placeholder('float', [None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    y_ = tf.placeholder('float', [None, 10])

    y_conv,train_step = model_LeNet(x_image,y_)
    accuracy = accuracy_cal(y_,y_conv)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    mnist_data_set = input_data.read_data_sets("MNIST_data/",one_hot=True)

    c = []
    start_time = time.time();print('start_time:',start_time)
    for i in range(200):
        batch_xs,batch_ys = mnist_data_set.train.next_batch(200)

        if i % 2 == 0 :
            train_accuracy = accuracy.eval(feed_dict={x:batch_xs,y_:batch_ys})
            print("step %d,training accuracy %g"%(i,train_accuracy))
            c.append(train_accuracy)
            #计算时间间隔
            end_time = time.time()
            print('time:',(end_time-start_time))
            start_time = end_time

        #训练数据
        train_step.run(feed_dict={x:batch_xs,y_:batch_ys})

    sess.close()
    plt.plot(c)
    plt.tight_layout()
    plt.savefig('LeNet3-png.png',dpi=200)

if __name__ == '__main__':
     main()