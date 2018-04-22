from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import os
import pickle
import re
import csv
from itertools import islice
import matplotlib.pyplot as plt


np.set_printoptions(threshold= 10000)
pattern = '\d'


def loadModelNN(modelpath):

    solver = loadDatafrPKL(modelpath)
    return solver

def loadDatafromFolderINX(img_folder, start, end):
    test_data = []

    for i in range(start, end + 1):
        exactPath = os.path.join(img_folder, '{}.jpg'.format(i))

        im = Image.open(exactPath)
        im = im.convert('L')

        im = np.array(im, dtype = np.float32)
        test_data.append(im)

    return np.array(test_data, dtyep = np.float32)

def loadDatafromFolder(img_folder):

    test_data = []

    for image in os.listdir(img_folder):
        path = os.path.join(img_folder, image)

        im = Image.open(path)
        im = im.convert('L')

        im = np.array(im, dtype = np.float32)
        test_data.append(im)

    return np.array(test_data, dtype = np.float32)

def dataPrepocessing(data):
    for i in range(data.shape[0]):
        data[i] = Zerocenter_ZCA_whitening_Global_Contrast_Normalize(data[i])
    return data

def dataLoad(path, start, end):

    data = []
    for i in range(start, end + 1):
        exactPath = "%s\%s" % (path, '{}.jpg'.format(i))

        im = Image.open(exactPath)
        im = im.convert('L')

        im = np.array(im, dtype = np.float32)
        data.append(im)

    return np.array(data, dtype=np.float32)

def loadfromCSV(path):

    x_train = []
    y_train = []
    x_eval = []
    y_eval = []
    x_test = []
    y_test = []

    data_file = csv.reader(open(path, 'r'))
    for row in islice(data_file, 1, None):
        type = row[2]
        if type == 'Training':
            y_train.append(int(row[0]))
            tmp = row[1].split()
            x_train.append(np.array(tmp,dtype=np.int32).reshape(48,48))
        elif type == 'PublicTest':
            y_eval.append(int(row[0]))
            tmp = row[1].split()
            x_eval.append(np.array(tmp, dtype=np.int32).reshape(48, 48))
        else:
            y_test.append(int(row[0]))
            tmp = row[1].split()
            x_test.append(np.array(tmp, dtype=np.int32).reshape(48, 48))

    return np.array(x_train, dtype=np.float32), np.array(y_train), np.array(x_eval, dtype=np.float32), np.array(y_eval), np.array(x_test, dtype=np.float32), np.array(y_test)


def saveData(DIC, filename):
    pickle.dump(DIC, open(filename, 'wb'))


def loadDatafrPKL(filename):
    DATADIC = pickle.load(open(filename, 'rb'))
    return DATADIC

def loadLabels(filename):

    y_train = []
    y_test = []

    with open(filename, 'r') as labels:
        while True:
            lines = labels.readline()
            if not lines:
                break
            if len(re.findall(pattern, lines)) == 0:
                continue
            p_tmp, l = [str for str in lines.split(',')]
            testortrain, tmp = [str for str in lines.split('/')]

            if testortrain == 'Train':
                y_train.append(int(l))
            else:
                y_test.append(int(l))

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return y_train, y_test


def plot_cm(cm):

    norm_conf = []
    for i in cm:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.coolwarm,
                    interpolation='nearest')

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    ax.tick_params(axis='both', which='major', labelsize=9)

    cb = fig.colorbar(res)
    alphabet = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    plt.xticks(range(width), alphabet[:width], )
    plt.yticks(range(height), alphabet[:height])
    plt.savefig('confusion_matrix.png', format='png')



def Zerocenter_ZCA_whitening_Global_Contrast_Normalize(data):
    data2 = ZeroCenter(data)
    data3 = zca_whitening(flatten_matrix(data2)).reshape(48, 48)
    data4 = global_contrast_normalize(data3)
    data5 = np.rot90(data4, 3)
    return data5

def ZeroCenter(data):
    data = data - np.mean(data, axis=0)
    return data

def flatten_matrix(matrix):
    vector = matrix.flatten(1)
    vector = vector.reshape(1, len(vector))
    return vector

def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=True,
                              sqrt_bias=10, min_divisor=1e-8):
    """
    __author__ = "David Warde-Farley"
    __copyright__ = "Copyright 2012, Universite de Montreal"
    __credits__ = ["David Warde-Farley"]
    __license__ = "3-clause BSD"
    __email__ = "wardefar@iro"
    __maintainer__ = "David Warde-Farley"
    .. [1] A. Coates, H. Lee and A. Ng. "An Analysis of Single-Layer
       Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
       http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    """
    assert X.ndim == 2, "X.ndim must be 2"
    scale = float(scale)
    assert scale >= min_divisor

    mean = X.mean(axis=1)
    if subtract_mean:
        X = X - mean[:, np.newaxis]
    else:
        X = X.copy()
    if use_std:
        ddof = 1
        if X.shape[1] == 1:
            ddof = 0
        normalizers = np.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = np.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale
    normalizers[normalizers < min_divisor] = 1.
    X /= normalizers[:, np.newaxis]  # Does not make a copy.
    return X

def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T) / inputs.shape[1]  # Correlation matrix
    U, S, V = np.linalg.svd(sigma)  # Singular Value Decomposition
    epsilon = 0.1  # Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(np.diag(S) + epsilon))), U.T)  # ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)  # Data whitening


def loadTrainData(path):

    DATA = loadDatafrPKL(path)
    train_data = DATA['x_train']
    train_labels = DATA['y_train']
    eval_data = DATA['x_test']
    eval_labels = DATA['y_test']


    train_data, eval_data = train_data / 255.0, eval_data / 255.0

    train_data = dataPrepocessing(train_data)
    eval_data =dataPrepocessing(eval_data)

    train_data = train_data.reshape(-1, 48 , 48 , 1)
    eval_data = eval_data.reshape(-1, 48, 48, 1)


    #train_labels = (np.arange(7) == train_labels[:, None]).astype(np.float32)
    #eval_labels = (np.arange(7) == eval_labels[:, None]).astype(np.float32)


    return train_data, train_labels, eval_data, eval_labels


def loadWholeData(path):

    DATA = loadDatafrPKL(path)
    train_data = DATA['x_train']
    train_labels = DATA['y_train']
    eval_data = DATA['x_test']
    eval_labels = DATA['y_test']

    x_train = np.append(train_data, eval_data, axis = 0)
    y_train = np.append(train_labels, eval_labels)

    x_train /= 255.0

    x_train = dataPrepocessing(x_train)

    x_train = x_train.reshape(-1, 48, 48, 1)

    return x_train, y_train


def loadTestData():
    DATA = loadDatafrPKL('D:\CourseWork\CNN\DATA_CSV.pkl')
    test_data = DATA['x_test']
    test_data /= 255.0
    test_labels = DATA['y_test']

    test_data = dataPrepocessing(test_data)
    test_data = test_data.reshape(-1, 48, 48, 1)

    test_data = test_data.astype(dtype= np.float32)
    test_labels = test_labels.astype(dtype = np.int32)

    #test_labels = (np.arange(7) == test_labels[:, None]).astype(np.float32)

    return test_data, test_labels