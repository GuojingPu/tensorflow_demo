#!/usr/bin/python3
# -*- coding:utf-8 -*-
import  pickle
import  numpy as np

import os

"""
www.cs.toronto.edu/~kriz/cifar.html
"""
print(os.path.join(os.path.dirname(os.path.realpath(__file__)),''))

def unipickle(file):
    with open(file,'rb') as f:
        dict = pickle.load(f,encoding='bytes')
    return  dict


def onehot(lables):
    len_lables = len(lables)
    n_class = max(lables) + 1
    print(n_class)
    onehot_lables = np.zeros((len_lables,n_class))
    print(onehot_lables)

    onehot_lables[np.arange(len_lables),lables] = 1

    return onehot_lables

file_1 = r'/Users/pgj/work/tensorfolw_demo/data/cifar-10-batches-bin/data_batch_1.bin'
file_2 = r'/Users/pgj/work/tensorfolw_demo/data/cifar-10-batches-bin/data_batch_2.bin'
file_3 = r'/Users/pgj/work/tensorfolw_demo/data/cifar-10-batches-bin/data_batch_3.bin'
file_4 = r'/Users/pgj/work/tensorfolw_demo/data/cifar-10-batches-bin/data_batch_4.bin'
file_5 = r'/Users/pgj/work/tensorfolw_demo/data/cifar-10-batches-bin/data_batch_5.bin'


data_1 = unipickle(file_1)
data_2 = unipickle(file_2)
data_3 = unipickle(file_3)
data_4 = unipickle(file_4)
data_5 = unipickle(file_5)


print(data_1)
x_train = np.concatenate((data_1['data'],data_2['data'],data_3['data'],data_4['data'],data_5['data']),axis=0)


