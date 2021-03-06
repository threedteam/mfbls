'''
@to do:save hog,hsv,kmeans,conv feature on SVHN
@link:
'''

from skimage.feature import hog
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import cv2
import os
import numpy as np
import tensorflow as tf
import time
from GetTrainValTestData import getData


# keras.__version__
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'# use cpu

'''hog'''
hog_shape=900

def get_img_hog(img):
    img = cv2.resize(img, (32 * 3, 32 * 3))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_hog = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
    return img_hog

def get_imgs_hogs(imgs):
    print('enter methond:get_imgs_hogs')
    imgs_hogs = np.zeros(shape=(imgs.shape[0], hog_shape))
    i=0
    for img in imgs:
        img_hog = get_img_hog(img)
        imgs_hogs[i,:]=img_hog
        i+=1
    # print(imgs_hogs.shape)
    print('leave methond:get_imgs_hogs')
    return imgs_hogs

'''hsv'''
hsvCH_shape = 96
hsvCM_shape = 9
hsvs_shape = 105

def get_img_hsvColorHistogram(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # 1,image;2,channel;3,mask;4,histSize;5,pixel range
    img_hsvCH = cv2.calcHist([img], [0, 1, 2], None, [6, 4, 4], [0, 180, 0, 256, 0, 256])

    # 1.input array;2.output array
    img_hsvCH = cv2.normalize(img_hsvCH, img_hsvCH).flatten()
    # print(img_hsvCH.shape)
    return img_hsvCH

def get_imgs_hsvColorHistograms(imgs):
    print('enter methond:get_imgs_hsvColorHistograms')
    imgs_hsvCHs = np.zeros(shape=(imgs.shape[0], hsvCH_shape))
    i = 0
    for img in imgs:
        img_hsvCH = get_img_hsvColorHistogram(img)
        imgs_hsvCHs[i, :] = img_hsvCH
        i += 1

    # print("imgs_hsvCHs.shape:",imgs_hsvCHs.shape)
    print('leave methond:get_imgs_hsvColorHistograms')
    return imgs_hsvCHs

def get_img_hsvColorMoment(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsvCM = np.zeros((9,))
    for ch in range(0, 3):
        img_ch = img[:, :, ch] / 255.
        img_hsvCM[ch] = np.mean(img_ch)  # 0,1,2存储一阶颜色矩
        img_hsvCM[3 + ch] = np.sqrt(np.mean(np.square(img_ch - img_hsvCM[ch])))  # 3,4,5存储二阶颜色矩
        img_hsvCM[6 + ch] = np.cbrt(np.mean(np.power(img_ch - img_hsvCM[ch], 3)))  # 6,7,8存储三阶颜色矩
    # print('img_hsvCM:',img_hsvCM)
    return img_hsvCM

def get_imgs_hsvColorMoments(imgs):
    print('enter methond:get_imgs_hsvColorMoments')
    imgs_hsvCMs = np.zeros(shape=(imgs.shape[0], hsvCM_shape))
    i = 0
    for img in imgs:
        img_hsvCM = get_img_hsvColorMoment(img)
        imgs_hsvCMs[i, :] = img_hsvCM
        i += 1
    # print("imgs_hsvCMs.shape:",imgs_hsvCMs.shape)
    print('leave methond:get_imgs_hsvColorMoments')
    return imgs_hsvCMs

'''kmeans'''

'''#1 channel'''
rfSize = 8
CIFAR_DIM = [32, 32, 3]
numPatches = 490000
numCentroids = 500
num_iters = 50

def getEstimator(imgs):
    ##############imgs--->imgs-patches ##################
    timestart = time.time()
    patches = []
    for i in range(numPatches):
        if (np.mod(i, 10000) == 0):
            print("sampling for Kmeans", i, "/", numPatches)
        start_r = np.random.randint(CIFAR_DIM[0] - rfSize)
        start_c = np.random.randint(CIFAR_DIM[1] - rfSize)
        patch = np.array([])
        img = imgs[np.mod(i, imgs.shape[0])]
        patch = np.append(patch, img[start_r:start_r + rfSize, start_c:start_c + rfSize].ravel())
        patches.append(patch)
    patches = np.array(patches)
    timeP = time.time()
    print('The Total img-2-patches Time is : ', timeP - timestart, ' seconds')
    # ##############normalization ##################
    patches = (patches - patches.mean(1)[:, None]) / np.sqrt(patches.var(1) + 10)[:, None]
    timeN = time.time()
    print('The Total normalization Time is : ', timeN - timeP, ' seconds')

    ################whiten###################################
    # eig:计算矩阵的特征值(E)以及特征向量(V)
    [E, V] = np.linalg.eig(np.cov(patches, rowvar=False))  # False/0,每一列代表一个变量，而行包含观察值
    P = V.dot(np.diag(np.sqrt(1 / (E + 0.1)))).dot(V.T)
    patches = patches.dot(P)
    timeW = time.time()
    print('The whiten Time is : ', timeW - timeN, ' seconds')
    ################use kmeans###################################
    # 构造聚类器
    estimator = MiniBatchKMeans(n_clusters=numCentroids, init='k-means++', max_iter=num_iters, batch_size=1000,
                                n_init=2)
    estimator.fit(patches)

    timeD = time.time()
    print('The kmeans computing Time is : ', timeD - timeW, ' seconds')
    return estimator, P

def sliding(img, window=[8, 8]):
    row = img.shape[0]
    col = img.shape[1]
    col_extent = col - window[1] + 1
    row_extent = row - window[0] + 1
    out = np.array([])

    start_idx = np.arange(window[0])[:, None] * col + np.arange(window[1])
    offset_idx = np.arange(row_extent)[:, None] * col + np.arange(col_extent)
    if len(out) == 0:
        out = np.take(img, start_idx.ravel()[:, None] + offset_idx.ravel())
    else:
        out = np.append(out, np.take(img, start_idx.ravel()[:, None] + offset_idx.ravel()), axis=0)
    return out.T

def extract_features(imgs, estimator, P):
    imgs_features = []
    rs = CIFAR_DIM[0] - rfSize + 1
    cs = CIFAR_DIM[1] - rfSize + 1
    idx = 0
    for img in imgs:
        idx += 1
        if not np.mod(idx, 1000):
            print("extract features", idx, '/', len(imgs))
        patches = sliding(img, [rfSize, rfSize])
        # #normalize & whiten
        patches = (patches - patches.mean(1)[:, None]) / (np.sqrt(patches.var(1) + 10)[:, None])
        # or:patches=preprocessing.scale(patches, axis=1)
        patches = patches.dot(P)
        # calculate distance
        dist = estimator.transform(patches)
        u = dist.mean(1)
        patches = np.maximum(-dist + u[:, None], 0)
        patches = np.reshape(patches, [rs, cs, -1])
        # sum pool
        q = []
        q.append(patches[0:rs // 2, 0:cs // 2].sum(0).sum(0))
        q.append(patches[0:rs // 2, cs // 2:cs - 1].sum(0).sum(0))
        q.append(patches[rs // 2:rs - 1, 0:cs // 2].sum(0).sum(0))
        q.append(patches[rs // 2:rs - 1, cs // 2:cs - 1].sum(0).sum(0))
        q = np.array(q).ravel()
        imgs_features.append(q)
    imgs_features = np.array(imgs_features)

    return imgs_features

'''conv'''

kernel_number_para = 0.2
pca_para=0.99

def get_kernel_g(f1Number, f2Number, f3Number, f4Number):
    kernel_g1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, f1Number], stddev=0.1))
    kernel_g2 = tf.Variable(tf.truncated_normal(shape=[3, 3, f1Number, f2Number], stddev=0.1))
    kernel_g3 = tf.Variable(tf.truncated_normal(shape=[3, 3, f2Number, f3Number], stddev=0.1))
    kernel_g4 = tf.Variable(tf.truncated_normal(shape=[4, 4, f3Number, f4Number], stddev=0.1))
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        kernel_mix1 = sess.run(kernel_g1)
        kernel_mix2 = sess.run(kernel_g2)
        kernel_mix3 = sess.run(kernel_g3)
        kernel_mix4 = sess.run(kernel_g4)
    return kernel_mix1, kernel_mix2, kernel_mix3, kernel_mix4

def get_conv_paras(x_in_number, kernel_number_para):
    f1Number = int(kernel_number_para * x_in_number / (32 * 32))  #
    f2Number = int(kernel_number_para * x_in_number / (16 * 16))  #
    f3Number = int(kernel_number_para * x_in_number / (8 * 8))  #
    f4Number = int(kernel_number_para * x_in_number / (4 * 4))  #
    kernel1, kernel2, kernel3, kernel4 = get_kernel_g(f1Number, f2Number, f3Number, f4Number)
    weights = {
        'conv1': kernel1,
        'conv2': kernel2,
        'conv3': kernel3,
        'conv4': kernel4
    }
    return weights

def get_bias_paras(x_in_number, kernel_number_para):
    f1Number = int(kernel_number_para * x_in_number / (32 * 32))
    f2Number = int(kernel_number_para * x_in_number / (16 * 16))
    f3Number = int(kernel_number_para * x_in_number / (8 * 8))
    f4Number = int(kernel_number_para * x_in_number / (4 * 4))  #

    biases = {
        'conv1_bias': tf.Variable(tf.constant(value=0.1, shape=[f1Number]), name="conv1bias" + np.str(f1Number)),
        'conv2_bias': tf.Variable(tf.constant(value=0.1, shape=[f2Number]), name="conv2bias" + np.str(f2Number)),
        'conv3_bias': tf.Variable(tf.constant(value=0.1, shape=[f3Number]), name="conv3bias" + np.str(f3Number)),
        'conv4_bias': tf.Variable(tf.constant(value=0.1, shape=[f4Number]), name="conv4bias" + np.str(f4Number))
    }
    # transform the type to numpy from tensor for the weights and biases not are changed after(using get_convlayers_out method)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        biases_new = sess.run(biases)
    return biases_new

def conv2d(x, W, padding_mode):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding=padding_mode, )

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def get_convlayers_out_batch(x, conv_weights, conv_biases):
    convpool4_out_x=[]
    batch_size = 10000
    times = x.shape[0] // batch_size

    for i in range(times):
        batch_imgs=x[i*batch_size:(i+1)*batch_size]
        temp=get_convlayers_once(batch_imgs,conv_weights,conv_biases)
        convpool4_out_x.extend(temp)

    #rest
    rest_imgs=x[times*batch_size:x.shape[0]]
    temp = get_convlayers_once(rest_imgs, conv_weights, conv_biases)
    convpool4_out_x.extend(temp)
    convpool4_out_x=np.array(convpool4_out_x)
    return convpool4_out_x

def get_convlayers_once(x, conv_weights, conv_biases):
    x = conv2d(x, conv_weights['conv1'], 'SAME')
    x = max_pool(x)
    x = tf.nn.relu(x + conv_biases['conv1_bias'])
    x = conv2d(x, conv_weights['conv2'], 'SAME')
    x = max_pool(x)
    x = tf.nn.relu(x + conv_biases['conv2_bias'])
    x = conv2d(x, conv_weights['conv3'], 'SAME')
    x = max_pool(x)
    x = tf.nn.relu(x + conv_biases['conv3_bias'])

    x = conv2d(x, conv_weights['conv4'], 'SAME')
    x = max_pool(x)
    h_pool4 = tf.nn.relu(x + conv_biases['conv4_bias'])
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        convpool4_out = sess.run(h_pool4)
    return convpool4_out

def get_convlayers_out(x, conv_weights, conv_biases):  # the type of return is Tensor
    if(x.shape[0]>=10000):
        convpool4_out_x=get_convlayers_out_batch(x, conv_weights, conv_biases)
        return convpool4_out_x
    else:
        convpool4_out_x=get_convlayers_once(x, conv_weights, conv_biases)
        return convpool4_out_x

'''save feas'''
def saveHsvFea(x_train,x_val,x_test):
    '''x_train'''
    time1 = time.time()
    x_train_hsvCHs = get_imgs_hsvColorHistograms(x_train)
    x_train_hsvCMs = get_imgs_hsvColorMoments(x_train)
    scaler_hsv = preprocessing.MinMaxScaler(feature_range=(0, 1))  # (0,1)scale
    x_train_hsvCMs = scaler_hsv.fit_transform(x_train_hsvCMs)
    x_train_hsv = np.hstack([x_train_hsvCHs, x_train_hsvCMs])  # hsvs
    time2 = time.time()
    print('X_train:The extracting hsv feature time is : ', time2 - time1, ' seconds')

    '''x_val'''
    time2 = time.time()
    x_val_hsvCHs = get_imgs_hsvColorHistograms(x_val)
    x_val_hsvCMs = get_imgs_hsvColorMoments(x_val)
    x_val_hsvCMs = scaler_hsv.transform(x_val_hsvCMs)
    x_val_hsv = np.hstack([x_val_hsvCHs, x_val_hsvCMs])  # hsvs
    time3 = time.time()
    print('X_val:The extracting hsv feature time is : ', time3 - time2, ' seconds')

    '''x_test'''
    time3 = time.time()
    x_test_hsvCHs = get_imgs_hsvColorHistograms(x_test)
    x_test_hsvCMs = get_imgs_hsvColorMoments(x_test)
    x_test_hsvCMs = scaler_hsv.transform(x_test_hsvCMs)
    x_test_hsv = np.hstack([x_test_hsvCHs, x_test_hsvCMs])  # hsvs
    time4 = time.time()
    print('X_test:The extracting hsv feature time is : ', time4 - time3, ' seconds')

    '''保存所有特征'''
    np.save("../result_svhn/x_train_hsvs.npy", x_train_hsv)
    np.save("../result_svhn/x_val_hsvs.npy", x_val_hsv)
    np.save("../result_svhn/x_test_hsvs.npy", x_test_hsv)

    print('x_train_hsv shape:', x_train_hsv.shape)  #hsv
    return x_train_hsv,x_val_hsv,x_test_hsv

def saveHogFea(x_train,x_val,x_test):
    '''x_train'''
    time1 = time.time()
    x_train_hog = get_imgs_hogs(x_train)#hog
    time2 = time.time()
    print('X_train:The extracting hog feature time is : ', time2 - time1, ' seconds')

    '''x_val'''
    time2 = time.time()
    x_val_hog = get_imgs_hogs(x_val)#hog
    time3 = time.time()
    print('X_val:The extracting hog feature time is : ', time3 - time2, ' seconds')

    '''x_test'''
    time3 = time.time()
    x_test_hog = get_imgs_hogs(x_test)#hog
    time4 = time.time()
    print('X_test:The extracting hog feature time is : ', time4 - time3, ' seconds')

    '''保存所有特征'''
    np.save("../result_svhn/x_train_hog.npy", x_train_hog)
    np.save("../result_svhn/x_val_hog.npy", x_val_hog)
    np.save("../result_svhn/x_test_hog.npy", x_test_hog)

    print('x_train_hog shape:',x_train_hog.shape)#hog
    return x_train_hog,x_val_hog,x_test_hog

def saveKmeansFea(x_train,x_val,x_test):
    '''x_train'''
    time1 = time.time()
    x_train = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_train]
    x_train = np.array(x_train)
    estimator, P = getEstimator(x_train)# get kmeans dictionary
    x_train_kmeans = extract_features(x_train, estimator, P)
    scaler_kmeans = preprocessing.MinMaxScaler(feature_range=(0, 1))
    x_train_kmeans = scaler_kmeans.fit_transform(x_train_kmeans)#k-means
    time2 = time.time()
    print('X_train:The extracting K-means feature time is : ', time2 - time1, ' seconds')

    '''x_val'''
    time2 = time.time()
    x_val = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_val]
    x_val = np.array(x_val)
    x_val_kmeans = extract_features(x_val, estimator, P)
    x_val_kmeans = scaler_kmeans.transform(x_val_kmeans)#k-means
    time3 = time.time()
    print('X_val:The extracting K-means feature time is : ', time3 - time2, ' seconds')

    '''x_test'''
    time3 = time.time()
    x_test = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_test]
    x_test = np.array(x_test)
    x_test_kmeans = extract_features(x_test, estimator, P)
    x_test_kmeans = scaler_kmeans.transform(x_test_kmeans)#k-means
    time4 = time.time()
    print('X_test:The extracting K-means  feature time is : ', time4 - time3, ' seconds')

    '''保存所有特征'''
    np.save("../result_svhn/x_train_kmeans.npy", x_train_kmeans)
    np.save("../result_svhn/x_val_kmeans.npy", x_val_kmeans)
    np.save("../result_svhn/x_test_kmeans.npy", x_test_kmeans)

    print('x_train_kmeans shape:',x_train_kmeans.shape)#k-means
    return x_train_kmeans,x_val_kmeans,x_test_kmeans

def saveConvFea(x_train, x_val, x_test, kernel_number_para,pca_para):

    '''preprocessing'''
    x_train = np.array(x_train, dtype='float32')
    x_val = np.array(x_val, dtype='float32')
    x_test = np.array(x_test, dtype='float32')
    x_train = x_train / 127.5 - 1.
    x_val = x_val / 127.5 - 1.
    x_test = x_test / 127.5 - 1.

    '''conv kernel'''
    x_train_num = x_train.shape[0]
    conv_weights = get_conv_paras(x_train_num, kernel_number_para)
    conv_biases = get_bias_paras(x_train_num, kernel_number_para)

    '''x_train'''
    time1 = time.time()
    convpool4_out = get_convlayers_out(x_train, conv_weights, conv_biases)  # , conv_weights, conv_biases)
    x_train_conv = convpool4_out.reshape(x_train_num, -1)
    # pca
    pca_conv = PCA(n_components=pca_para)
    x_train_conv = pca_conv.fit_transform(x_train_conv)

    time2 = time.time()
    print('X_train:The extracting conv feature time is : ', time2 - time1, ' seconds')

    '''x_val'''
    time2 = time.time()
    convpool4_outVAl = get_convlayers_out(x_val, conv_weights, conv_biases)  # conv_weights, conv_biases
    x_val_conv = convpool4_outVAl.reshape(x_val.shape[0], -1)
    x_val_conv = pca_conv.transform(x_val_conv)
    time3 = time.time()
    print('X_val:The extracting conv feature time is : ', time3 - time2, ' seconds')

    '''x_test'''
    time3 = time.time()
    convpool4_outTest = get_convlayers_out(x_test, conv_weights, conv_biases)  # conv_weights, conv_biases
    x_test_conv = convpool4_outTest.reshape(x_test.shape[0], -1)
    x_test_conv = pca_conv.transform(x_test_conv)
    time4 = time.time()
    print('X_test:The extracting conv feature time is : ', time4 - time3, ' seconds')

    '''保存所有特征'''
    np.save("../result_svhn/x_train_conv.npy", x_train_conv)
    np.save("../result_svhn/x_val_conv.npy", x_val_conv)
    np.save("../result_svhn/x_test_conv.npy", x_test_conv)
    print('x_train_conv shape:', x_train_conv.shape)
    return x_train_conv,x_val_conv,x_test_conv


if __name__ == '__main__':

    timeS = time.time()

    x_train, y_train, x_val, y_val, x_test, y_test=getData()
    print('x_train shape:',x_train.shape)
    print('x_val shape:', x_val.shape)
    print('x_test shape:', x_test.shape)

    # save feas of svhn
    saveHsvFea(x_train, x_val, x_test)
    saveHogFea(x_train, x_val, x_test)
    saveKmeansFea(x_train, x_val, x_test)
    saveConvFea(x_train, x_val, x_test, kernel_number_para, pca_para)
