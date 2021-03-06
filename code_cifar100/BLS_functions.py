'''
@to do:bls related functions.
@link:Most of the code in the file comes from http://broadlearning.ai/bls/
'''

from __future__ import print_function, division
import os
import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA
import time




'''
#输出训练/测试准确率
'''


def show_accuracy(predictLabel, Label):
    count = 0
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return (round(count / len(Label), 5))


'''
激活函数
'''


def tansig(x):
    return (2 / (1 + np.exp(-2 * x))) - 1


def sigmoid(data):
    return 1.0 / (1 + np.exp(-data))


def linear(data):
    return data


def tanh(data):
    return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))


def relu(data):
    return np.maximum(data, 0)


def pinv(A, reg):
    return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)


'''
参数压缩
'''


def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z


'''
参数稀疏化
'''


def sparse_bls(A, b):
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def softmax(x):
    x = np.array(x)
    x -= np.max(x, axis=1, keepdims=True) 
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return x
