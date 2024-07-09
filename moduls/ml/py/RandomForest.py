#!/usr/bin/env python
# coding: utf-8

from random import choice

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
# from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import to_categorical


from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
# from sklearn.externals import joblib
from sklearn.metrics import r2_score
import joblib


# the Configurable parameters of RF
class Param:
    """


    """
    n_estimators = 10
    criterion = 'gini'
    # criterion = 'mse'
    max_depth = None
    min_samples_split = 2
    min_samples_leaf = 1
    min_weight_fraction_leaf = 0.0
    max_features = 'auto'
    max_leaf_nodes = None
    min_impurity_decrease = 0.0
    min_impurity_split = None
    bootstrap = True
    oob_score = False
    # n_jobs = None
    n_jobs = 1
    random_state = 42
    verbose = 0
    warm_start = False
    class_weight = None
    ccp_alpha = 0
    max_samples = None


class Option:
    sample_weight = None
    check_input = True
    X_idx_sorted = None


# def runAll(x_train, x_test, y_train, y_test, params, Option, savePath=None, typ="RandomForestClassifier"):
def RF_Model(x_train, y_train, params, typ="RandomForestClassifier"):
    # 参数准备
    param_use = Param
    param_use.max_depth = params[0]
    param_use.max_leaf_nodes = params[1]
    param_use.n_estimators = params[2]
    param_use = param_process(param_use)

    model = model_select(param_use, typ)

    # x_train, x_val, y_train, y_val = create_dataSet(path)
    # trainParam = get_trainParam(x_train, y_train, Option)
    model = train_model(x_train, y_train, model)
    # savePath = save_model(model, savePath)
    return model


# After the interface is configured as 0, it can be recognized as None when converted to python
def param_process(Param):
    if Param.max_depth == 0:
        Param.max_depth = None
    if Param.random_state == 0:
        Param.random_state = None
    if Param.max_leaf_nodes == 0:
        Param.max_leaf_nodes = None
    if Param.min_impurity_split == 0:
        Param.min_impurity_split = None
    if Param.n_jobs == 0:
        Param.n_jobs = None
    if Param.max_samples == 0:
        Param.max_samples = None

    else:
        Param.class_weight = 'balanced'
    return Param


def option_process(Option):
    if Option.sample_weight == 0:
        Option.sample_weight = None
    if Option.X_idx_sorted == 0:
        Option.X_idx_sorted = None
    return Option


# choose classifier or regressor
def model_select(Param, typ="RandomForestClassifier"):
    if typ == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=Param.n_estimators,
                                       max_depth=Param.max_depth,
                                       min_samples_split=Param.min_samples_split,
                                       min_samples_leaf=Param.min_samples_leaf,
                                       max_features=Param.max_features, random_state=Param.random_state,
                                       max_leaf_nodes=Param.max_leaf_nodes,
                                       min_impurity_decrease=Param.min_impurity_decrease,
                                       # min_impurity_split=Param.min_impurity_split,
                                       bootstrap=Param.bootstrap, oob_score=Param.oob_score, n_jobs=Param.n_jobs,
                                       verbose=Param.verbose,
                                       warm_start=Param.warm_start, class_weight=Param.class_weight)
        return model
    else:
        model = RandomForestRegressor(n_estimators=Param.n_estimators,
                                      max_depth=Param.max_depth,
                                      min_samples_split=Param.min_samples_split,
                                      min_samples_leaf=Param.min_samples_leaf,
                                      max_features=Param.max_features, random_state=Param.random_state,
                                      max_leaf_nodes=Param.max_leaf_nodes,
                                      min_impurity_decrease=Param.min_impurity_decrease,
                                      # min_impurity_split=Param.min_impurity_split,
                                      bootstrap=Param.bootstrap, oob_score=Param.oob_score, n_jobs=Param.n_jobs,
                                      verbose=Param.verbose,
                                      warm_start=Param.warm_start)
        # class_weight=Param.class_weight
        return model


def get_trainParam(Xtrain, Ytrain, Option):
    trainParam = {'data': Xtrain, 'label': Ytrain, 'sample_weight': Option.sample_weight,
                  'check_input': Option.check_input, 'X_idx_sorted': Option.X_idx_sorted}
    return trainParam


def train_model(x_train, y_train, model):
    # sample_weight = None
    # check_input = True
    # X_idx_sorted = None
    y_train = y_train.T[0]
    model = model.fit(x_train, y_train, sample_weight=None)
    return model


def get_score(x_train, x_test, y_train, y_test, params, rul_pre=False):

    if not rul_pre:
        rf = RF_Model(x_train, y_train, params, typ="RandomForestClassifier")
        pre = rf.predict(x_test)
        # 正确率
        score = rf.score(x_test, y_test)

    else:
        rf = RF_Model(x_train, y_train, params, typ="RandomForestRegressor")
        pre = np.array(rf.predict(x_test).tolist())
        y_test_use = y_test.ravel()
        score = r2_score(y_test_use, pre)  # 准确率

    # Error = 1 - score
    return pre, score


def create_dataSet(path):
    list = []
    path1 = path[0]
    path2 = path[1]
    feature = 'feature'
    for i in range(0, 576):  # 给每一列特征加上名字column='feature'+i
        column = feature + str(i)
        list.append(column)
    df = pd.read_csv(path1, names=list, header=0)  # 读取训练集并且给训练集的特征重命名
    # 训练集
    DataSet = df[:11000].iloc[:, 0:576]  # 取前110000行，所有列的数据
    df1 = pd.read_csv(path2)  # 读取训练集的标签 drive/MyDrive/2020WS/GA/
    df1.columns = ['label']  # 给训练集的特征重命名
    LabelSet = df1[:11000].iloc[:, :]  # 取前110000行的标签
    # 将label转化为独热编码
    Labels = to_categorical(LabelSet, num_classes=None)
    X_train, X_val, Y_train, Y_val = train_test_split(DataSet, Labels, test_size=0.3, random_state=42)  # 将30%的数据集变为验证集
    x_train = X_train.values
    x_val = X_val.values

    y_train = Y_train
    y_val = Y_val
    return x_train, x_val, y_train, y_val


def fun_score(savePath, path):
    model = load_model(savePath)
    x_train, x_val, y_train, y_val = create_dataSet(path)
    score = model.score(x_val, y_val)
    return score


def fun_predict(savePath, path):
    model = load_model(savePath)
    x_val = create_dataSet(path)[1]
    array_list = model.predict(x_val).tolist()
    length = len(array_list)
    prediction = []
    for i in range(0, length):
        label = array_list[i]
    for j in range(0, len(label)):
        if label[j] != 0:
            prediction.append(j + 1)

    return prediction


def save_model(model, savePath):
    joblib.dump(model, savePath)
    return savePath


def load_model(savePath):
    model = joblib.load(savePath)
    return model


if __name__ == '__main__':
    pass
    # path1 = '/content/drive/MyDrive/2020WS/GA/traindata.csv'
    # path2 = '/content/drive/MyDrive/2020WS/GA/trainlabel.csv'
    # save_path = '/content/drive/MyDrive/2020WS/GA/test3.pkl'
    # selection = [4, 20, 1, 12, 8, 30, 5]
    # threshold = 0.95
    # best_generation, dna_set, original_pop, original_finess = run(3, 5, 0.5, 0.5, 7, selection, [path1, path2], "RandomForestClassifier", save_path, threshold)
    #
    # # 运行模型
    # runAll(path, typ, Param1, Option1, savePath)
    # # 打印成绩
    # score = RandomForest.fun_score(savePath, path)

    # print(best_generation)
    # print(dna_set)
    # print(original_pop)
    # print(original_finess)
    #
    # # fitness_figure(best_generation)
