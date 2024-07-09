#!/usr/bin/env python
# coding: utf-8


import joblib
import tensorflow as tf
# import keras
# from keras import layers
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from random import choice

from moduls.ml.dataset import import_data


class AutoencoderParam:
    units = 100
    activation = None
    use_bias = False
    kernel_initializer = 'glorot_uniform'
    bias_initializer = 'zeros'
    kernel_regularizer = None
    bias_regularizer = None
    activity_regularizer = None
    kernel_constraint = None
    bias_constraint = None


class evaluateParam:
    x = None
    y = None
    batch_size = None
    verbose = 1
    sample_weight = None
    steps = None


class predictParam:
    # x
    batch_size = None
    verbose = 0
    steps = None


def get_data(path):
    train_data = np.array(pd.read_csv(path))  # 读取训练数据
    test_data = np.array(pd.read_csv(path))  # 读取测试数据
    shape = train_data.shape[1]
    return shape, train_data, test_data


# 主函数
def train_model(x_train, x_test, y_train, y_test, results, rul_pre=False):
    # results[LayerCount,units1,units2,units3,epochs,batchSize,denseActivation,optimizer,loss]
    tf.random.set_seed(42)
    Shape = x_train.shape
    print(Shape)
    print(y_train.shape)
    input_img = Input(shape=Shape[1])
    n = results[0]
    encoded = decoded = None
    for i in range(n):
        if i == 0:
            encoded = Dense(results[i + 1], activation=results[6])(input_img)
        else:
            encoded = Dense(results[i + 1], activation=results[6])(encoded)

        for j in range(n):
            if j == n - 1:
                if not rul_pre:
                    decoded = Dense(y_train.shape[1], activation='softmax')(encoded)
                else:
                    decoded = Dense(y_train.shape[1], activation='sigmoid')(encoded)
                    # decoded = Dense(4, activation=choice(['softmax', 'sigmoid', 'relu', 'tanh']))(encoded)
            else:
                decoded = Dense(results[n - j - 1], activation=results[6])(encoded)

    autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.compile(optimizer=results[7], loss=results[8])
    autoencoder.summary()
    model = autoencoder.fit(x_train, y_train,
                            epochs=results[4],
                            batch_size=results[5],
                            validation_data=(x_test, y_test)
                            )
    #
    return model


# 函数评分
def get_accuracy(x_train, x_test, y_train, y_test, results, rul_pre=False, savePath=None):

    hist = train_model(x_train, x_test, y_train, y_test, results, rul_pre=rul_pre)

    pre = hist.model.predict(x_test)

    if not rul_pre:
        pre_use = np.argmax(pre, axis=1)
        y_test_use = np.argmax(y_test, axis=1)
        score = accuracy_score(y_test_use, pre_use)  # 准确率
    else:
        score = r2_score(y_test, pre)
        pre_use = pre
        y_test_use = y_test

    Error = 1 - score

    y = (y_test_use, pre_use)

    return y, Error


def y_transfer(y_list):

    y_list = np.argmin(y_list, axis=1)
    return y_list
    # y = []
    # for list_i in y_list:
    #     max_value = max(list_i)
    #     idx = list_i.index(max_value) + 1
    #     y.append(idx)
    # return np.array(y)

def evaluate_model(savePath, path, evaluateParam):
    model = load_model(savePath)
    data = np.array(pd.read_csv(path))
    evaluate_result = model.evaluate(evaluateParam)
    return evaluate_result


def predict(savePath, path, predictParam):
    model = load_model(savePath)
    data = np.array(pd.read_csv(path))
    prediction = model.predict(predictParam)
    return prediction


def save_model(model, savePath):
    joblib.dump(model, savePath)
    return savePath


def load_model(savePath):
    model = joblib.load(savePath)
    return model


if __name__ == '__main__':
    data_path = './Dataset/'
    save_path = './'
    x_train, x_test, y_train, y_test = import_data(data_path, model='SVM')

    # train_data = x_train
    # test_data = x_test
    # results[LayerCount, units1, units2, units3, epochs, batchSize, denseActivation, optimizer, loss]
    # LayerCount[1, 4], int
    # units1[210, 250], int
    # units2[160, 209], int
    # units3[100, 155], int
    # epochs[20, 20], int
    # batchSize[128, 128], int
    # denseActivation['softmax', 'sigmoid', 'relu', 'tanh']
    # optimizer['adam', 'rmsprop', 'adagrad']
    # loss['mse', 'mae', 'categorical_crossentropy']

    results = [2, 220, 200, 120, 20, 128, 'relu', 'adam', 'mse']
    # results = [2, 220, 200, 120, 2, 128, 'relu', 'adam', 'mse']
    # Model = train_data(train_data, test_data, results)
    res = get_accuracy(x_train, x_test, y_train, y_test, results)
    print("最终评分为：", res)

    # [1, 220, 189, 124, 20, 128, 'softmax', 'adam', 'categorical_crossentropy']
