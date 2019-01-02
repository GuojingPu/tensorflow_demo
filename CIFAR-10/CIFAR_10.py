#!/usr/bin/python3
# -*- coding:utf-8 -*-
import  pickle
import  numpy as np
import tensorflow as tf
import time
import os
import matplotlib
matplotlib.use('TkAgg')
import  matplotlib.pyplot as plt

"""
www.cs.toronto.edu/~kriz/cifar.html
"""
print(os.path.join(os.path.dirname(os.path.realpath(__file__)),''))


learning_rate = 0.001
training_iters = 200
batch_size = 50
display_step = 5
n_features = 32 * 32 * 3
n_classes= 10
n_fc1 = 384
n_fc2 = 192




def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

        print(type(dict))
    return dict


def onehot(lables):
    n_sample = len(lables)
    n_class = max(lables) + 1
    print(n_sample,n_class)
    onehot_lables = np.zeros((n_sample,n_class))
    onehot_lables[np.arange(n_sample),lables] = 1
    return onehot_lables



def weight_variable(shape,stddev):
    initial = tf.truncated_normal(shape,stddev=stddev)
    variable = tf.Variable(initial)
    return variable


def bias_varible(value,shape):
    initial = tf.constant(value,dtype=tf.float32,shape=shape)
    varible=tf.Variable(initial)
    return varible


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")



def avg_pool_3x3(value):
    return tf.nn.avg_pool(value=value,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def lrn(input):
    return tf.nn.lrn(input,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75)

def load_data():

    dir =  r'/Users/pgj/work/tensorfolw_demo/data/cifar-10-batches-py'
    dir =  r'D:\work\tensorflow_demo\data\cifar-10-batches-py'

    file_1 =  dir + r'/data_batch_1'
    file_2 =  dir + r'/data_batch_2'
    file_3 =  dir + r'/data_batch_3'
    file_4 =  dir + r'/data_batch_4'
    file_5 =  dir + r'/data_batch_5'

    file_6 = dir + r'/test_batch'

    data_1 = unpickle(file_1)
    data_2 = unpickle(file_2)
    data_3 = unpickle(file_3)
    data_4 = unpickle(file_4)
    data_5 = unpickle(file_5)

    test_data = unpickle(file_6)

    # print(data_1)
    x_train = np.concatenate((data_1[b'data'],data_2[b'data'],data_3[b'data'],data_4[b'data'],data_5[b'data']),axis=0)

    y_train = np.concatenate((data_1[b'labels'],data_2[b'labels'],data_3[b'labels'],data_4[b'labels'],data_5[b'labels']),axis=0)
    y_train = onehot(y_train)
    print(y_train.shape,y_train)


    # print(test_data)
    x_test = test_data[b'data'][:5000,:]
    y_test = test_data[b'labels'][:5000]

    print(x_test.shape,len(y_test))


    return (x_train,y_train,x_test,y_test)




def model(x,y):

    W_conv = {
        'conv1':weight_variable([5,5,3,32],stddev=0.0001),
        'conv2':weight_variable([5,5,32,64],stddev=0.01),
        'fc1':weight_variable([8*8*64,384],stddev=0.1),
        'fc2':weight_variable([384,192],stddev=0.1),
        'fc3':weight_variable([192,10],stddev=0.1),
    }

    b_conv = {
        'conv1':bias_varible(0.0,shape=[32]),
        'conv2':bias_varible(0.1,shape=[64]),
        'fc1':bias_varible(0.1,shape=[384]),
        'fc2':bias_varible(0.1,shape=[192]),
        'fc3':bias_varible(0.0,shape=[10])
    }


    x_image = tf.reshape(x,[-1,32,32,3])

    #卷积层1
    conv1 = conv2d(x_image,W_conv['conv1'])
    conv1 = tf.nn.bias_add(conv1,b_conv['conv1'])
    conv1 =  tf.nn.relu(conv1)

    #池化层1
    pool1 = avg_pool_3x3(conv1)

    #LRN函数
    norm1 = lrn(pool1)

    #卷积层2
    conv2 = conv2d(norm1,W_conv['conv2'])
    conv2 = tf.nn.bias_add(conv2,b_conv['conv2'])
    conv2 = tf.nn.relu(conv2)

    #LRN
    norm2 = lrn(conv2)

    #池化层2
    # pool2 = avg_pool_3x3(norm2)
    pool2 = max_pool_2x2(norm2)

    reshape = tf.reshape(pool2,[-1,8 * 8 * 64])

    #输出层1
    fc1 = tf.add(tf.matmul(reshape,W_conv['fc1']),b_conv['fc1'])
    fc1 = tf.nn.relu(fc1)

    # 输出层2
    fc2 = tf.add(tf.matmul(fc1,W_conv['fc2']),b_conv['fc2'])
    fc2 = tf.nn.relu(fc2)

    #输出层3
    fc3 = tf.nn.softmax(tf.add(tf.matmul(fc2,W_conv['fc3']),b_conv['fc3']))

    print(fc3.shape,y)
    #损失函数
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=fc3,logits=y))
    loss = -tf.reduce_mean(y * tf.log(fc3))  # 损失函数/熵

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    conditions = tf.equal(tf.argmax(fc3,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(conditions,tf.float32))

    return  y,optimizer,accuracy



def main():
    x = tf.placeholder(tf.float32, [None, n_features])
    y = tf.placeholder(tf.float32, [None, n_classes])


    y,optimizer,accuracy=model(x,y)

    x_train, y_train, x_test, y_test = load_data()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        c = []
        total_batch = int(x_train.shape[0] / batch_size) #50000/50  x_train (50000,3072)

        start_time = time.time()
        # for i in range(200):
        for batch in range(total_batch):
            batch_x = x_train[batch * batch_size : (batch+1) * batch_size, :]
            batch_y = y_train[batch * batch_size : (batch+1) * batch_size, :]

            sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})

            train_accuracy = accuracy.eval(feed_dict={x: batch_x, y: batch_y})
            print("step %d,training accuracy %g" % (batch, train_accuracy))
            c.append(train_accuracy)
          # 计算时间间隔
            end_time = time.time()
            print('time:', (end_time - start_time))
            start_time = end_time

        # acc = sess.run(accuracy,feed_dict={x:x_test,y:y_test})
        # print("test accuary ",acc)
        plt.plot(c)
        plt.xlabel("Iter")
        plt.ylabel("Cost")
        # plt.title('lr=%f,ti=%d,bs=%d,acc=%f'%(learning_rate,total_batch,batch_size,acc))
        plt.tight_layout()
        plt.savefig('CIFAR-10-png.png', dpi=200)


if __name__ == '__main__':
    main()