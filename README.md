# Proposed Model(MFBLS)

## Introduce
A multi-feature broad learning system (MFBLS) that aims to improve the image classification performance of broad learning system (BLS) and its variants is proposed. The model is characterized by two major characteristics: multi-feature extraction and parallel structure. MFBLS extract multiple features, such as convolutional feature, K-means feature, HOG feature, and color feature, to improve the performance of classification.  Besides, a parallel architecture that is suitable for multi-feature extraction is proposed for MSBLS. There are four feature blocks and one fusion block in this structure. The extracted features are used directly as the feature nodes in feature block, hence no random feature mapping is needed for feature nodes generation. In addition, a “stacking with ridge regression” strategy is applied to the fusion block in the structure to get the final output of MFBLS. Experimental results show that the proposed model can achieve the accuracies of 92.25%, 81.03%, and 54.66% for SVHN, CIFAR-10, and CIFAR-100 respectively.

## Code for Propose Model

SVHN: subdirectory code_svhn.

CIFAR-10: subdirectory code_cifar10

CIFAR-100: subdirectory code_cifar100

ImFea_svhn.py, ImFea_10.py, and ImFea_100.py are used to extract four types of features on SVHN, CIFAR10, and CIFAR100, respectively.

## Runtime Environment 
Intel Xeon E5-2678 CPU with 128G memory

python 3.6

keras 2.2.4

## Public dataset

### SVHN

http://ufldl.stanford.edu/housenumbers/

### CIFAR-10 & CIFAR-100

http://www.cs.toronto.edu/~kriz/cifar.html
