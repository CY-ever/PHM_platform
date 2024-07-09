import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from moduls.feature_selection.load import data_load
import os
import scipy.io as sio
import pandas as pd
from moduls.feature_selection.writein import writein
from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func


def PCA_main(data, method=1,dimension_method=0,dimension=5,percent=95,save_path='./',output_file=0):
    '''

    :param data: input features,mxn
    :param switch: switch: 0:features in raw,1:features in column
    :param method: 0:特征值分解；1：奇异值分解
    :param dimension_method: int,0,1,2;降维方式
    :param dimension: int,dimension after reduction
    :param percent: float,percent of PCA to remain 降维后保留的主成分百分比,输入为百分号前的数值
    :param save_path: path to save
    :param output_file: type to save image,0:png,1:jpg,2:svg,3:pdf
    :return:
    '''
    X = data
    dim_limit = data.shape[1]

    if method==0:
        # 计算特征值和特征向量
        if dimension > 0 and dimension <= dim_limit:
            eigvalue, eigvector = np.linalg.eig(np.cov(X.T))
            index= np.argsort(-eigvalue)# 按降序排列

            list=[]
            for i in range(dimension):
                eigvector_new=eigvector[:,index[i]]
                list.append(eigvector_new)
            array=np.array(list)
            array=array.T

            X = X-X.mean(axis=0)

            X_new = X.dot(array)
            X_new=np.real(X_new)
        else:
            abort(400, "The feature dimension must be in [1, %d]." % dim_limit)
    elif method==1:
        if dimension_method==0:
            if dimension > 0 and dimension <= min(data.shape[0],data.shape[1]):
                pca = PCA(n_components=dimension) #选择降到多少维度
            else:
                abort(400, "The feature dimension must be between 0 and min(n_samples, n_features).")
        elif dimension_method== 1:
            if percent>0 and percent<100:
                pca = PCA(n_components=percent / 100)  # 选择降维后的数据所保留的百分比
            else:
                abort(400, "The percentage must be in (0, 100].")
        elif dimension_method==2:
            if data.shape[0]>= data.shape[1]:
                pca=PCA(n_components='mle') # MLE算法自己选择降维维度的效果
            else:
                abort(400,"n_components='mle' is only supported if n_samples >= n_features.")
        pca.fit(X)
        # 降维后的数据
        X_new = pca.transform(X)
        X_new = np.real(X_new)
    # 保存数据
    save_data_func(data=X_new, output_file=output_file, save_path=save_path,
                   file_name="PCA",
                   index_label="Downscaled data")

    return X_new


if __name__ == "__main__":
    data = writein('traindata.mat',1)
    X_new=PCA_main(data)
    print(X_new.shape)




