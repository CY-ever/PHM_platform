import numpy as np
import matplotlib.pyplot as plt
from moduls.feature_selection.load import data_load
import os
import scipy.io as sio
import pandas as pd
from moduls.feature_selection.writein import writein

from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func


def FDA(data, labels, dim=2,save_path='./',output_file=0):
    '''

    :param data: features,mxn,二维数组
    :param labels: labels,mx1，二维数组
    :param switch: 0:features in raw,1:features in column
    :param dim: select dimension after reduction
    :param save_path:path to save
    :param output_file: type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :param output_image: type to save image,0:png,1:jpg,2:svg,3:pdf
    :return:
    '''
    if labels is None:
        abort(400,"The label file is missing.")
    y=np.squeeze(labels)
    x=data

    clusters = np.unique(y)
    if dim>x.shape[1] or dim <1:
        abort(400,'The feature dimension must be in [1, %d].'% x.shape[1])
    if len(clusters) == 2:
        x_1 = np.array([x[i] for i in range(len(x)) if y[i] == 0])
        x_2 = np.array([x[i] for i in range(len(x)) if y[i] == 1])

        mju1 = np.mean(x_1, axis=0)     # mean vector
        mju2 = np.mean(x_2, axis=0)

        sw1 = np.dot((x_1 - mju1).T, (x_1 - mju1))    # Within-class scatter matrix
        sw2 = np.dot((x_2 - mju2).T, (x_2 - mju2))
        sw = sw1 + sw2
        A=(mju1 - mju2).reshape(-1, 1).T
        w=np.linalg.inv(sw)*A
        w_new=w[:,:dim]
        data_ndim=np.dot(x,w_new) #降维之后的数据
        data_ndim = np.real(data_ndim)
        # 保存数据
        save_data_func(data=data_ndim, output_file=output_file, save_path=save_path,
                       file_name="FDA",
                       index_label="Downscaled data")

        return data_ndim

    elif len(clusters) > 2:

            # within_class scatter matrix
        Sw = np.zeros((data.shape[1], data.shape[1]))
        for i in clusters:
            # Swi = np.zeros((data.shape[1], data.shape[1]))
            datai = data[y == i]
            datai = datai - datai.mean(0)
            Swi = np.mat(datai).T * np.mat(datai)
            Sw += Swi

        # between_class scatter matrix
        SB = np.zeros((data.shape[1], data.shape[1]))
        u = data.mean(0)  # 所有样本的平均值
        for i in clusters:
            Ni = data[y == i].shape[0]
            ui = data[y == i].mean(0)  # 某个类别的平均值
            SBi = Ni * np.mat(ui - u).T * np.mat(ui - u)
            SB += SBi

        S = np.linalg.inv(Sw) * (SB)
        eigVals, eigVects = np.linalg.eig(S)  # 求特征值，特征向量
        eigValInd = np.argsort(eigVals)
        eigValInd = eigValInd[:(-dim - 1):-1]
        w = eigVects[:, eigValInd]
        data_ndim = np.dot(data, w)
        data_ndim=np.real(data_ndim)
        # 保存数据
        save_data_func(data=data_ndim, output_file=output_file, save_path=save_path,
                       file_name="FDA",
                       index_label="Downscaled data")

        return data_ndim

if __name__ =="__main__":
    #测试外圈内圈时域特征数据结果
    features = writein('outer_inner_time_features.mat',1)
    labels = writein('outer_inner_labels.mat',1)
    x = features
    y = labels
    data_dim = FDA(x, y,output_file=3)
    print(y.shape)
    print(data_dim.shape)



