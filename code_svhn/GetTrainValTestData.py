'''
@to do:construct a training set and a validation set from the training set and extra set of svhn, .
@link:
'''

from keras.utils.np_utils import to_categorical
import os
import numpy as np
import time
from scipy.io import loadmat

# keras.__version__
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'# use cpu

'''proprecessing the data'''
# refer:blog.csdn.net/juezhanangle/article/details/73203693
def reformat(x, y):
    # 改变原始数据的形状
    # (图片高，图片宽，通道数，图片数)->(图片数,图片高，图片宽，通道数)
    x = np.transpose(x, (3, 0, 1, 2))
    # y 变成one-hot encoding,y:change 10 to 0
    y[y == 10] = 0
    y = to_categorical(y, 10)
    y = y.astype(np.float64)
    return x, y

'''get_train_val'''
def get_train_val(x_train,y_train, x_extra,y_extra):
    x_temp=np.concatenate((x_train,x_extra),axis=0)
    y_temp=np.concatenate((y_train,y_extra),axis=0)

    train_n=110000
    val_n=1000
    x_train=x_temp[0:train_n]
    y_train = y_temp[0:train_n]
    x_val=x_temp[train_n:train_n+val_n]
    y_val=y_temp[train_n:train_n+val_n]
    return x_train, y_train, x_val, y_val

'''get x_train, y_train, x_val, y_val, x_test, y_test'''
def getData():
    # 数据载入
    traindata = loadmat('../data/train.mat')
    extradata = loadmat('../data/extra.mat')
    testdata = loadmat('../data/test.mat')

    x_train = traindata['X']# X(32,32,3,73257)
    y_train = traindata['y']#,Y(73257,1)
    x_extra = extradata['X']# X(32,32,3,531131)
    y_extra = extradata['y']#Y(531131,1)
    x_test = testdata['X']# X(32,32,3,26032)
    y_test = testdata['y']#Y(26032,1)

    # reformat
    x_train, y_train = reformat(x_train, y_train)# X(73257,32,32,3),Y(73257,10)
    x_extra, y_extra = reformat(x_extra, y_extra)# X(531131,32,32,3),Y(531131,10)
    x_test, y_test = reformat(x_test, y_test)# X(26032,32,32,3),Y(26032,10)

    # get train,val
    x_train, y_train, x_val, y_val = get_train_val(x_train, y_train, x_extra, y_extra) # x_train:110000   ,x_val:1000

    return  x_train, y_train, x_val, y_val,x_test, y_test


if __name__ == '__main__':

    timeS = time.time()

    x_train, y_train, x_val, y_val, x_test, y_test=getData()
    print('x_train shape:',x_train.shape)
    print('x_val shape:', x_val.shape)
    print('x_test shape:', x_test.shape)

