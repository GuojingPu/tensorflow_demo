#!/usr/bin/python3
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time


"""****************************** API介绍 *************************************"""
"""

    tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32,
                        seed=None, name=None)
    从截断的正态分布中输出随机值。生成的值服从具有指定平均值和标准偏差的正态分布，
    如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
    shape: 一维的张量，也是输出的张量。
    mean: 正态分布的均值。 
    stddev: 正态分布的标准差。
    dtype: 输出的类型。
    seed: 一个整数，当设置之后，每次生成的随机数都一样。
    name: 操作的名字
"""

"""
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

从正态分布中输出随机值。
参数:
    shape: 一维的张量，也是输出的张量。
    mean: 正态分布的均值。
    stddev: 正态分布的标准差。
    dtype: 输出的类型。
    seed: 一个整数，当设置之后，每次生成的随机数都一样。
    name: 操作的名字。

"""

"""
tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None) 
均匀分布随机数，范围为[minval,maxval]
"""


"""
    tf.constant(value, dtype=None, shape=None, name=’Const’) 
    创建一个常量tensor，按照给出value来赋值，可以用shape来指定其形状。
    value可以是一个数，也可以是一个list。 如果是一个数，那么这个常亮中
    所有值的按该数来赋值。 如果是list,那么len(value)一定要小于等于shape
    展开后的长度。赋值时，先将value中的值逐个存入。不够的部分，则全部存
    入value的最后一个值。
"""

"""
tf.reshape
tf.reshape(tensor, shape, name=None) 
顾名思义，就是将tensor按照新的shape重新排列。一般来说，shape有三种用法： 
如果 shape=[-1], 表示要将tensor展开成一个list 
如果 shape=[a,b,c,…] 其中每个a,b,c,..均>0，那么就是常规用法 
如果 shape=[a,-1,c,…] 此时b=-1，a,c,..依然>0。这表示tf会根据tensor的原尺寸，自动计算b的值。 
"""


"""
tf.nn.dropout
dropout(x, keep_prob, noise_shape=None, seed=None, name=None) 
按概率来将x中的一些元素值置零，并将其他的值放大。用于进行dropout操作，
一定程度上可以防止过拟合 .x是一个张量，而keep_prob是一个（0,1]之间的值。
x中的各个元素清零的概率互相独立，为1-keep_prob,而没有清零的元素，则会统
一乘以1/keep_prob, 目的是为了保持x的整体期望值不变。

"""

"""
tf.nn.conv2d (input, filter, strides, padding, use_cudnn_on_gpu=None, 
                                    data_format=None, name=None)

input :  输入的要做卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_weight, in_channel ]，
         其中batch为图片的数量，in_height 为图片高度，in_weight 为图片宽度，in_channel 为图片的通道数，
         灰度图该值为1，彩色图为3。（也可以用其它值，但是具体含义不是很理解）
filter： 卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，
         其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 
         的 in_channel 要保持一致，out_channel 是卷积核数量。
strides：卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
padding：string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。"SAME"是考虑边界，不足的时候用0
         去填充周围，"VALID"则不考虑
use_cudnn_on_gpu：bool类型，是否使用cudnn加速，默认为true

"""
"""*******************************************************************"""




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



x = tf.placeholder('float',[None,784])
y_ = tf.placeholder('float',[None,10])

x_image = tf.reshape(x,[-1,28,28,1])

print('x_image:',x_image)



"""1 卷积层1"""
#第一层卷积核
# filter1 = tf.Variable(tf.truncated_normal([5,5,1,6]))
filter1 = weight_variable([5,5,1,6])

#第一层输出误差
bias1 = tf.Variable(tf.truncated_normal([6]))


print('**',filter1,bias1)

#第一次卷积运算
conv1 = tf.nn.conv2d(x_image,filter1,strides=[1,1,1,1],padding="SAME")
print('conv1:',conv1)

"""2 激活层1"""
# h_conv1 = tf.nn.sigmoid(conv1 + bias1)
h_conv1 = tf.nn.relu(conv1 + bias1)
print('h_conv1:',h_conv1)

"""3 池化层2"""
maxPool2 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

print('maxPool2:',maxPool2)

"4 卷积层2"
filter2 = tf.Variable(tf.truncated_normal([5,5,6,16]))
#第二层输出误差
bias2 = tf.Variable(tf.truncated_normal([16]))
print('**',filter2,bias2)

#第二次卷积运算
conv2 = tf.nn.conv2d(maxPool2,filter2,strides=[1,1,1,1],padding="SAME")
print('conv2:',conv2)

"""5 激活层2 """
# h_conv2 = tf.nn.sigmoid(conv2 + bias2)
h_conv2 = tf.nn.relu(conv2 + bias2)
print('h_conv2:',h_conv2)

"""6 池化层3"""
maxPool3 = tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
print('maxPool3:',maxPool3)


"7 卷积层3"
filter3 = tf.Variable(tf.truncated_normal([5,5,16,120]))
#第三层输出误差
bias3 = tf.Variable(tf.truncated_normal([120]))
print('**',filter3,bias3)
#卷积运算
conv3 = tf.nn.conv2d(maxPool3,filter3,strides=[1,1,1,1],padding="SAME")
print('conv2:',conv2)

"""8 激活层3"""
# h_conv3 = tf.nn.sigmoid(conv3 + bias3)
h_conv3 = tf.nn.relu(conv3 + bias3)
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
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
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

#损失函数/熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

#使用梯度下降算法对模型进行训练
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)


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

