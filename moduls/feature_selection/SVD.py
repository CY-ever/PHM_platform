import numpy as np
from moduls.feature_selection.load import data_load
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import pandas as pd
from moduls.feature_selection.writein import writein
from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func

def SVD(data,dimension=2,save_path='./',output_file=0):
    '''
    :param data: input signal,mxn
    :param switch:int,switch=1: n is numbers of features，特征在列；switch=0: m is numbers of features,特征在行
    :param dimension: dimension after reduction
    :param save_path: path to save
    :param output_file: type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :param output_image: type to save image,0:png,1:jpg,2:svg,3:pdf
    :return:
    '''
    U, sigma, VT = np.linalg.svd(data)
    # 默认特征在列的情况
    if dimension>VT.shape[0] or dimension <1:
        abort(400,'The feature dimension must be in [1, %d].' % VT.shape[0])
    else:
        VT_new = VT[:dimension, :]
        data_new = np.dot(VT_new, np.mat(data).T).T
        data_new = data_new.A
    # 保存数据
    save_data_func(data=data_new, output_file=output_file, save_path=save_path,
                   file_name="SVD",
                   index_label="Downscaled data")

    return data_new


if __name__== "__main__":
    # data=data_load('time_features.mat', "time_features", ".mat", "row")
    data = writein('1_1_seg_time_features.mat',1)
    data_new=SVD(data)
    print(data_new.shape)
    print(data.shape)

