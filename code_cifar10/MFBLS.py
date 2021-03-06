'''
@to do:using hsv+hog+kmeans+conv features to classify the CIFAR10 datasets
@link:
@link:
'''

from __future__ import print_function, division
import os
from keras.utils.np_utils import to_categorical
from keras.datasets import cifar10
from BLS_functions import *
from ImFea_10 import saveHsvFea,saveHogFea,saveKmeansFea,saveConvFea

'''Related parameters.Note that these parameters need to be tuned'''

# conv
conv_c = 0.001
conv_s = 0.85
conv_e = 10000
kernel_number_para = 0.18
pca_para=0.99

# kmeans
kmeans_c = 0.001
kmeans_s = 0.7
kmeans_e = 10000

#hog
hog_c = 0.005
hog_s = 0.7
hog_e = 10000

#hsvs
hsvs_c = 0.001
hsvs_s = 0.85
hsvs_e = 5000

fusion_c = 1  # Regularization coefficient

def MFBLS(x_train, train_y, x_val, val_y, x_test, test_y):
    print('conv_c:', conv_c, 'conv_s:', conv_s, 'conv_e:', conv_e)
    print('kmeans_c:', kmeans_c, 'kmeans_s:', kmeans_s, 'kmeans_e:', kmeans_e)
    print('hog_c:', hog_c, 'hog_s:', hog_s, 'hog_e:', hog_e)
    print('hsvs_c:', hsvs_c, 'hsvs_s:', hsvs_s, 'hsvs_e:', hsvs_e)
    #
    # feat
    # conv
    x_train_conv = np.load("../result_cifar10/x_train_conv.npy")  # -train
    x_val_conv = np.load("../result_cifar10/x_val_conv.npy")  # -val
    x_test_conv = np.load("../result_cifar10/x_test_conv.npy")  # -testures

    # k_means
    x_train_kmeans = np.load("../result_cifar10/x_train_kmeans.npy")
    x_val_kmeans = np.load("../result_cifar10/x_val_kmeans.npy")
    x_test_kmeans = np.load("../result_cifar10/x_test_kmeans.npy")

    # # hog
    x_train_hog = np.load("../result_cifar10/x_train_hog.npy")
    x_val_hog = np.load("../result_cifar10/x_val_hog.npy")
    x_test_hog = np.load("../result_cifar10/x_test_hog.npy")

    #hsvs
    x_train_hsvs = np.load("../result_cifar10/x_train_hsvs.npy")
    x_val_hsvs = np.load("../result_cifar10/x_val_hsvs.npy")
    x_test_hsvs = np.load("../result_cifar10/x_test_hsvs.npy")

    conv_shape = x_train_conv.shape[1]
    kmeans_shape = x_train_kmeans.shape[1]
    hog_shape = x_train_hog.shape[1]
    hsvs_shape = x_train_hsvs.shape[1]

    time_start = time.time()  # ????????????

    # ???????????????1
    InOfEnhLayer1WithBias = np.hstack([x_train_conv, 0.1 * np.ones((x_train_conv.shape[0], 1))])
    if conv_shape >= conv_e:
        random.seed(67797325)
        weiOfEnhLayer1 = LA.orth(2 * random.randn(conv_shape + 1, conv_e)) - 1
    else:
        random.seed(67797325)
        weiOfEnhLayer1 = LA.orth(2 * random.randn(conv_shape + 1, conv_e).T - 1).T
    tempOfOutOfEnhLayer1 = np.dot(InOfEnhLayer1WithBias, weiOfEnhLayer1)
    parameterOfShrink1 = conv_s / np.max(tempOfOutOfEnhLayer1)
    OutOfEnhLayer1 = tansig(tempOfOutOfEnhLayer1 * parameterOfShrink1)

    # ??????C1
    InputOfC1Layer = np.hstack([x_train_conv, OutOfEnhLayer1])
    pinvOfInputC1 = pinv(InputOfC1Layer, conv_c)
    C1Weight = np.dot(pinvOfInputC1, train_y)
    OutC1 = np.dot(InputOfC1Layer, C1Weight)

    # ???????????????2
    # ?????????2??????
    InOfEnhLayer2WithBias = np.hstack([x_train_kmeans, 0.1 * np.ones((x_train_kmeans.shape[0], 1))])
    # ???????????????2??????
    if kmeans_shape >= kmeans_e:
        random.seed(67797325)
        weiOfEnhLayer2 = LA.orth(2 * random.randn(kmeans_shape + 1, kmeans_e)) - 1
    else:
        random.seed(67797325)
        weiOfEnhLayer2 = LA.orth(2 * random.randn(kmeans_shape + 1, kmeans_e).T - 1).T
    tempOfOutOfEnhLayer2 = np.dot(InOfEnhLayer2WithBias, weiOfEnhLayer2)
    parameterOfShrink2 = kmeans_s / np.max(tempOfOutOfEnhLayer2)
    OutOfEnhLayer2 = tansig(tempOfOutOfEnhLayer2 * parameterOfShrink2)

    # ??????C2
    InputOfC2Layer = np.hstack([x_train_kmeans, OutOfEnhLayer2])
    pinvOfInputC2 = pinv(InputOfC2Layer, kmeans_c)
    C2Weight = np.dot(pinvOfInputC2, train_y)
    OutC2 = np.dot(InputOfC2Layer, C2Weight)

    # ???????????????3
    # ?????????3??????
    InOfEnhLayer3WithBias = np.hstack([x_train_hog, 0.1 * np.ones((x_train_hog.shape[0], 1))])
    # ???????????????3??????
    if hog_shape >= hog_e:
        random.seed(67797325)
        weiOfEnhLayer3 = LA.orth(2 * random.randn(hog_shape + 1, hog_e)) - 1
    else:
        random.seed(67797325)
        weiOfEnhLayer3 = LA.orth(2 * random.randn(hog_shape + 1, hog_e).T - 1).T
    tempOfOutOfEnhLayer3 = np.dot(InOfEnhLayer3WithBias, weiOfEnhLayer3)
    parameterOfShrink3 = hog_s / np.max(tempOfOutOfEnhLayer3)
    OutOfEnhLayer3 = tansig(tempOfOutOfEnhLayer3 * parameterOfShrink3)

    # ??????C3
    InputOfC3Layer = np.hstack([x_train_hog, OutOfEnhLayer3])
    pinvOfInputC3 = pinv(InputOfC3Layer, hog_c)
    C3Weight = np.dot(pinvOfInputC3, train_y)
    OutC3 = np.dot(InputOfC3Layer, C3Weight)

    # ???????????????4
    # ?????????4??????
    InOfEnhLayer4WithBias = np.hstack([x_train_hsvs, 0.1 * np.ones((x_train_hsvs.shape[0], 1))])
    # ???????????????4??????
    if hsvs_shape >= hsvs_e:
        random.seed(67797325)
        weiOfEnhLayer4 = LA.orth(2 * random.randn(hsvs_shape + 1, hsvs_e)) - 1
    else:
        random.seed(67797325)
        weiOfEnhLayer4 = LA.orth(2 * random.randn(hsvs_shape + 1, hsvs_e).T - 1).T
    tempOfOutOfEnhLayer4 = np.dot(InOfEnhLayer4WithBias, weiOfEnhLayer4)
    parameterOfShrink4 = hsvs_s / np.max(tempOfOutOfEnhLayer4)
    OutOfEnhLayer4 = tansig(tempOfOutOfEnhLayer4 * parameterOfShrink4)

    # ??????C4
    InputOfC4Layer = np.hstack([x_train_hsvs, OutOfEnhLayer4])
    pinvOfInputC4 = pinv(InputOfC4Layer, hsvs_c)
    C4Weight = np.dot(pinvOfInputC4, train_y)
    OutC4 = np.dot(InputOfC4Layer, C4Weight)

    # normalize OutC1,OutC2,OutC3
    OutC1_N = softmax(OutC1)
    OutC2_N = softmax(OutC2)
    OutC3_N = softmax(OutC3)
    OutC4_N = softmax(OutC4)

    # ??????????????????
    InputOfOutputLayer = np.hstack([OutC1_N, OutC2_N, OutC3_N,OutC4_N])  #
    pinvOfInput = pinv(InputOfOutputLayer, fusion_c)
    OutputWeight = np.dot(pinvOfInput, train_y)  # ????????????
    time_end = time.time()  # ????????????
    trainTime = time_end - time_start

    # ????????????
    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain, train_y)
    print('Training accurate is', trainAcc * 100, '%')
    print('Training time is ', trainTime, 's')

    # val process
    time_start = time.time()  # ??????????????????

    #  ?????????
    InOfEnhLayer1WithBiasVal = np.hstack([x_val_conv, 0.1 * np.ones((x_val_conv.shape[0], 1))])
    tempOfOutOfEnhLayer1Val = np.dot(InOfEnhLayer1WithBiasVal, weiOfEnhLayer1)
    OutOfEnhLayer1Val = tansig(tempOfOutOfEnhLayer1Val * parameterOfShrink1)

    InOfEnhLayer2WithBiasVal = np.hstack([x_val_kmeans, 0.1 * np.ones((x_val_kmeans.shape[0], 1))])
    tempOfOutOfEnhLayer2Val = np.dot(InOfEnhLayer2WithBiasVal, weiOfEnhLayer2)
    OutOfEnhLayer2Val = tansig(tempOfOutOfEnhLayer2Val * parameterOfShrink2)

    InOfEnhLayer3WithBiasVal = np.hstack([x_val_hog, 0.1 * np.ones((x_val_hog.shape[0], 1))])
    tempOfOutOfEnhLayer3Val = np.dot(InOfEnhLayer3WithBiasVal, weiOfEnhLayer3)
    OutOfEnhLayer3Val = tansig(tempOfOutOfEnhLayer3Val * parameterOfShrink3)

    InOfEnhLayer4WithBiasVal = np.hstack([x_val_hsvs, 0.1 * np.ones((x_val_hsvs.shape[0], 1))])
    tempOfOutOfEnhLayer4Val = np.dot(InOfEnhLayer4WithBiasVal, weiOfEnhLayer4)
    OutOfEnhLayer4Val = tansig(tempOfOutOfEnhLayer4Val * parameterOfShrink4)

    # ??????C
    InputOfC1LayerVal = np.hstack([x_val_conv, OutOfEnhLayer1Val])
    OutC1Val = np.dot(InputOfC1LayerVal, C1Weight)

    InputOfC2LayerVal = np.hstack([x_val_kmeans, OutOfEnhLayer2Val])
    OutC2Val = np.dot(InputOfC2LayerVal, C2Weight)

    InputOfC3LayerVal = np.hstack([x_val_hog, OutOfEnhLayer3Val])
    OutC3Val = np.dot(InputOfC3LayerVal, C3Weight)

    InputOfC4LayerVal = np.hstack([x_val_hsvs, OutOfEnhLayer4Val])
    OutC4Val = np.dot(InputOfC4LayerVal, C4Weight)

    # normalize OutC1Val,OutC2Val,OutC3Val
    OutC1Val_N = softmax(OutC1Val)
    OutC2Val_N = softmax(OutC2Val)
    OutC3Val_N = softmax(OutC3Val)
    OutC4Val_N = softmax(OutC4Val)

    #  ???????????????
    InputOfOutputLayerVal = np.hstack([OutC1Val_N, OutC2Val_N, OutC3Val_N,OutC4Val_N])  #
    #  ??????????????????
    OutputOfVal = np.dot(InputOfOutputLayerVal, OutputWeight)
    time_end = time.time()  # ????????????
    valTime = time_end - time_start
    valAcc = show_accuracy(OutputOfVal, val_y)
    print('Val accurate is', valAcc * 100, '%')
    print('Val time is ', valTime, 's')

    # ????????????
    time_start = time.time()  # ??????????????????

    #  ?????????
    InOfEnhLayer1WithBiasTest = np.hstack([x_test_conv, 0.1 * np.ones((x_test_conv.shape[0], 1))])
    tempOfOutOfEnhLayer1Test = np.dot(InOfEnhLayer1WithBiasTest, weiOfEnhLayer1)
    OutOfEnhLayer1Test = tansig(tempOfOutOfEnhLayer1Test * parameterOfShrink1)

    InOfEnhLayer2WithBiasTest = np.hstack([x_test_kmeans, 0.1 * np.ones((x_test_kmeans.shape[0], 1))])
    tempOfOutOfEnhLayer2Test = np.dot(InOfEnhLayer2WithBiasTest, weiOfEnhLayer2)
    OutOfEnhLayer2Test = tansig(tempOfOutOfEnhLayer2Test * parameterOfShrink2)

    InOfEnhLayer3WithBiasTest = np.hstack([x_test_hog, 0.1 * np.ones((x_test_hog.shape[0], 1))])
    tempOfOutOfEnhLayer3Test = np.dot(InOfEnhLayer3WithBiasTest, weiOfEnhLayer3)
    OutOfEnhLayer3Test = tansig(tempOfOutOfEnhLayer3Test * parameterOfShrink3)

    InOfEnhLayer4WithBiasTest = np.hstack([x_test_hsvs, 0.1 * np.ones((x_test_hsvs.shape[0], 1))])
    tempOfOutOfEnhLayer4Test = np.dot(InOfEnhLayer4WithBiasTest, weiOfEnhLayer4)
    OutOfEnhLayer4Test = tansig(tempOfOutOfEnhLayer4Test * parameterOfShrink4)

    # ??????C
    InputOfC1LayerTest = np.hstack([x_test_conv, OutOfEnhLayer1Test])
    OutC1Test = np.dot(InputOfC1LayerTest, C1Weight)

    InputOfC2LayerTest = np.hstack([x_test_kmeans, OutOfEnhLayer2Test])
    OutC2Test = np.dot(InputOfC2LayerTest, C2Weight)

    InputOfC3LayerTest = np.hstack([x_test_hog, OutOfEnhLayer3Test])
    OutC3Test = np.dot(InputOfC3LayerTest, C3Weight)

    InputOfC4LayerTest = np.hstack([x_test_hsvs, OutOfEnhLayer4Test])
    OutC4Test = np.dot(InputOfC4LayerTest, C4Weight)

    # normalize OutC1,OutC2,OutC3
    OutC1Test_N = softmax(OutC1Test)
    OutC2Test_N = softmax(OutC2Test)
    OutC3Test_N = softmax(OutC3Test)
    OutC4Test_N = softmax(OutC4Test)

    #  ???????????????
    InputOfOutputLayerTest = np.hstack([OutC1Test_N, OutC2Test_N, OutC3Test_N,OutC4Test_N])  #

    #  ??????????????????
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    time_end = time.time()  # ????????????
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest, test_y)
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')
    return trainAcc,valAcc,testAcc


if __name__ == '__main__':

    # ????????????
    (X_train, Y_train), (x_test, y_test) = cifar10.load_data()
    x_train = X_train[0:49000, ]
    x_val = X_train[49000:50000, ]

    # ?????????????????????
    y_train = to_categorical(Y_train[0:49000, ], 10)
    y_val = to_categorical(Y_train[49000:50000, ], 10)
    y_test = to_categorical(y_test, 10)

    print('x_train shape:', x_train.shape[0])
    print('x_val shape:', x_val.shape[0])
    print('x_test shape:', x_test.shape[0])

    '''run'''
    print('================extract the feas =======================')
    saveHsvFea(x_train, x_val, x_test)
    saveHogFea(x_train, x_val, x_test)
    saveKmeansFea(x_train, x_val, x_test)
    saveConvFea(x_train, x_val, x_test, kernel_number_para, pca_para)

    print('================run mfbls=======================')
    MFBLS(x_train, y_train, x_val, y_val, x_test, y_test)

