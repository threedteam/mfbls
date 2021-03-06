'''
@to do:using hsv+hog+kmeans+conv features to classify the CIFAR100 datasets
@link:
@link:
'''

from __future__ import print_function, division
from keras.utils.np_utils import to_categorical
from keras.datasets import cifar100
from BLS_functions import *
from ImFea_100 import saveHsvFea,saveHogFea,saveKmeansFea,saveConvFea

############################

'''Related parameters.Note that these parameters need to be tuned'''
# conv
kernel_number_para = 0.18
pca_para=0.99
conv_c = 0.01
conv_s = 0.85
conv_e = 9900

# kmeans
kmeans_c = 0.01
kmeans_s = 0.95
kmeans_e = 9900

#hog
hog_c = 0.001
hog_s = 0.71
hog_e = 9500

#hsvs
hsvs_c = 0.001
hsvs_s = 0.95
hsvs_e = 9500

fusion_c = 0.005  # Regularization coefficient

def MFBLS(x_train, train_y, x_val, val_y, x_test, test_y):
    print('conv_c:', conv_c, 'conv_s:', conv_s, 'conv_e:', conv_e)
    print('kmeans_c:', kmeans_c, 'kmeans_s:', kmeans_s, 'kmeans_e:', kmeans_e)
    print('hog_c:', hog_c, 'hog_s:', hog_s, 'hog_e:', hog_e)
    print('hsvs_c:', hsvs_c, 'hsvs_s:', hsvs_s, 'hsvs_e:', hsvs_e)
    #
    # feat
    # conv
    x_train_conv = np.load("../result_cifar100/x_train_conv.npy")  # -train
    x_val_conv = np.load("../result_cifar100/x_val_conv.npy")  # -val
    x_test_conv = np.load("../result_cifar100/x_test_conv.npy")  # -testures
    # print(x_train_conv.shape)

    # k_means
    x_train_kmeans = np.load("../result_cifar100/x_train_kmeans.npy")
    x_val_kmeans = np.load("../result_cifar100/x_val_kmeans.npy")
    x_test_kmeans = np.load("../result_cifar100/x_test_kmeans.npy")

    # # hog
    x_train_hog = np.load("../result_cifar100/x_train_hog.npy")
    x_val_hog = np.load("../result_cifar100/x_val_hog.npy")
    x_test_hog = np.load("../result_cifar100/x_test_hog.npy")

    #hsvs
    x_train_hsvs = np.load("../result_cifar100/x_train_hsvs.npy")
    x_val_hsvs = np.load("../result_cifar100/x_val_hsvs.npy")
    x_test_hsvs = np.load("../result_cifar100/x_test_hsvs.npy")

    conv_shape = x_train_conv.shape[1]
    kmeans_shape = x_train_kmeans.shape[1]
    hog_shape = x_train_hog.shape[1]
    hsvs_shape = x_train_hsvs.shape[1]

    time_start = time.time()  # 计时开始

    # 生成强化层1
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

    # 生成C1
    InputOfC1Layer = np.hstack([x_train_conv, OutOfEnhLayer1])
    pinvOfInputC1 = pinv(InputOfC1Layer, conv_c)
    C1Weight = np.dot(pinvOfInputC1, train_y)
    OutC1 = np.dot(InputOfC1Layer, C1Weight)

    # 生成强化层2
    # 强化层2输入
    InOfEnhLayer2WithBias = np.hstack([x_train_kmeans, 0.1 * np.ones((x_train_kmeans.shape[0], 1))])
    # 生成强化层2权重
    if kmeans_shape >= kmeans_e:
        random.seed(67797325)
        weiOfEnhLayer2 = LA.orth(2 * random.randn(kmeans_shape + 1, kmeans_e)) - 1
    else:
        random.seed(67797325)
        weiOfEnhLayer2 = LA.orth(2 * random.randn(kmeans_shape + 1, kmeans_e).T - 1).T
    tempOfOutOfEnhLayer2 = np.dot(InOfEnhLayer2WithBias, weiOfEnhLayer2)
    parameterOfShrink2 = kmeans_s / np.max(tempOfOutOfEnhLayer2)
    OutOfEnhLayer2 = tansig(tempOfOutOfEnhLayer2 * parameterOfShrink2)

    # 生成C2
    InputOfC2Layer = np.hstack([x_train_kmeans, OutOfEnhLayer2])
    pinvOfInputC2 = pinv(InputOfC2Layer, kmeans_c)
    C2Weight = np.dot(pinvOfInputC2, train_y)
    OutC2 = np.dot(InputOfC2Layer, C2Weight)

    # 生成强化层3
    # 强化层3输入
    InOfEnhLayer3WithBias = np.hstack([x_train_hog, 0.1 * np.ones((x_train_hog.shape[0], 1))])
    # 生成强化层3权重
    if hog_shape >= hog_e:
        random.seed(67797325)
        weiOfEnhLayer3 = LA.orth(2 * random.randn(hog_shape + 1, hog_e)) - 1
    else:
        random.seed(67797325)
        weiOfEnhLayer3 = LA.orth(2 * random.randn(hog_shape + 1, hog_e).T - 1).T
    tempOfOutOfEnhLayer3 = np.dot(InOfEnhLayer3WithBias, weiOfEnhLayer3)
    parameterOfShrink3 = hog_s / np.max(tempOfOutOfEnhLayer3)
    OutOfEnhLayer3 = tansig(tempOfOutOfEnhLayer3 * parameterOfShrink3)

    # 生成C3
    InputOfC3Layer = np.hstack([x_train_hog, OutOfEnhLayer3])
    pinvOfInputC3 = pinv(InputOfC3Layer, hog_c)
    C3Weight = np.dot(pinvOfInputC3, train_y)
    OutC3 = np.dot(InputOfC3Layer, C3Weight)

    # 生成强化层4
    # 强化层4输入
    InOfEnhLayer4WithBias = np.hstack([x_train_hsvs, 0.1 * np.ones((x_train_hsvs.shape[0], 1))])
    # 生成强化层4权重
    if hsvs_shape >= hsvs_e:
        random.seed(67797325)
        weiOfEnhLayer4 = LA.orth(2 * random.randn(hsvs_shape + 1, hsvs_e)) - 1
    else:
        random.seed(67797325)
        weiOfEnhLayer4 = LA.orth(2 * random.randn(hsvs_shape + 1, hsvs_e).T - 1).T
    tempOfOutOfEnhLayer4 = np.dot(InOfEnhLayer4WithBias, weiOfEnhLayer4)
    parameterOfShrink4 = hsvs_s / np.max(tempOfOutOfEnhLayer4)
    OutOfEnhLayer4 = tansig(tempOfOutOfEnhLayer4 * parameterOfShrink4)

    # 生成C4
    InputOfC4Layer = np.hstack([x_train_hsvs, OutOfEnhLayer4])
    pinvOfInputC4 = pinv(InputOfC4Layer, hsvs_c)
    C4Weight = np.dot(pinvOfInputC4, train_y)
    OutC4 = np.dot(InputOfC4Layer, C4Weight)

    # normalize OutC1,OutC2,OutC3
    OutC1_N = softmax(OutC1)
    OutC2_N = softmax(OutC2)
    OutC3_N = softmax(OutC3)
    OutC4_N = softmax(OutC4)

    # 生成最终输入
    InputOfOutputLayer = np.hstack([OutC1_N, OutC2_N, OutC3_N,OutC4_N])  #
    pinvOfInput = pinv(InputOfOutputLayer, fusion_c)
    OutputWeight = np.dot(pinvOfInput, train_y)  # 全局违逆
    time_end = time.time()  # 训练完成
    trainTime = time_end - time_start

    # 训练输出
    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain, train_y)
    print('Training accurate is', trainAcc * 100, '%')
    print('Training time is ', trainTime, 's')

    # val process
    time_start = time.time()  # 测试计时开始

    #  强化层
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

    # 生成C
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

    #  最终层输入
    InputOfOutputLayerVal = np.hstack([OutC1Val_N, OutC2Val_N, OutC3Val_N,OutC4Val_N])  #
    #  最终测试输出
    OutputOfVal = np.dot(InputOfOutputLayerVal, OutputWeight)
    time_end = time.time()  # 训练完成
    valTime = time_end - time_start
    valAcc = show_accuracy(OutputOfVal, val_y)
    print('Val accurate is', valAcc * 100, '%')
    print('Val time is ', valTime, 's')

    # 测试过程
    time_start = time.time()  # 测试计时开始

    #  强化层
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

    # 生成C
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

    #  最终层输入
    InputOfOutputLayerTest = np.hstack([OutC1Test_N, OutC2Test_N, OutC3Test_N,OutC4Test_N])  #

    #  最终测试输出
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    time_end = time.time()  # 训练完成
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest, test_y)
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')
    return trainAcc,valAcc,testAcc


if __name__ == '__main__':

    # 数据载入
    (X_train, Y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    x_train = X_train[0:49000, ]
    x_val = X_train[49000:50000, ]

    # 多分类标签生成
    y_train = to_categorical(Y_train[0:49000, ], 100)#abc
    y_val = to_categorical(Y_train[49000:50000, ], 100)
    y_test = to_categorical(y_test, 100)

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


