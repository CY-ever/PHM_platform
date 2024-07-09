import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pywt
import scipy.io as sio
import os
from moduls.data_denoise.DWT.writein import writein
from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func


#sgn函数
def sgn(num):
    if(num > 0.0):
        return 1.0
    elif(num == 0.0):
        return 0.0
    else:
        return -1.0

def DWT_threshold_filter(new_df,name='db1',N=3,threshold_method=2,threshold_coeff=0.5,save_path='./',output_file=0,output_image=0):
    '''

    :param new_df: input domain signal,1-D,一维数组
    :param name: Select wavelet mother function
    :param N: number of wavelet decompositions
    :param threshold_method: 0：Soft Threshold，1：Hard Threshold，2:Intermediate Threshold
    :param threshold_coeff:coeff for Intermediate Threshold,between[0,1)
    :param save_path:path to save
    :param output_file: type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :param output_image: type to save image,0:png,1:jpg,2:svg,3:pdf
    :return:
    '''
    if N<1:
        abort(400, "ERROR: The level must be a positive integer.")
    data = new_df.tolist()  # 将np.ndarray()转为列表
    coeffs=pywt.wavedec(data, name, level=N)
    ca=coeffs[0]
    cd=coeffs[1:]
    # print(cd[0][60])
    length0 = len(data)

    Cd1 = np.array(coeffs[-1])
    abs_cd1 = np.abs(Cd1)
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0 ), math.e))#固定阈值计算
    usecoeffs = []
    usecoeffs.append(ca)  # 向列表末尾添加对象

    #软硬阈值折中的方法
    # a = 0.5
    for i in range(len(cd)):
        length=len(cd[i])
        for k in range(length):
            # 阈值选择：
            if threshold_method==0:
                # 软阈值
                if (abs(cd[i][k]) >= lamda):
                    cd[i][k] = sgn(cd[i][k]) * (abs(cd[i][k]) - lamda)
                else:
                    cd[i][k] = 0.0
            if threshold_method==1:
                # 硬阈值
                if (abs(cd[i][k]) >= lamda):
                    cd[i][k] = cd[i][k]
                else:
                    cd[i][k] = 0.0
            if threshold_method == 2:
                # 软硬阈值折中，threshold_coeff取值在[0,1]
                if threshold_coeff>=0 and threshold_coeff<=1:
                    if (abs(cd[i][k]) >= lamda):
                        cd[i][k] = sgn(cd[i][k]) * (abs(cd[i][k]) - threshold_coeff * lamda)
                    else:
                        cd[i][k] = 0.0
                else:
                    abort(400, "ERROR: The threshold-coeff must be in [0,1].")
        usecoeffs.append(cd[i])

    recoeffs = pywt.waverec(usecoeffs, name)  # 信号重构
    # 保存数据
    save_data_func(data=recoeffs, output_file=output_file, save_path=save_path,
                   file_name="DWT_threshold_filter",
                   index_label="Filtered signal")

    # plt.figure(figsize=(12, 5))
    plt.subplot(2, 1, 1)
    plt.plot(new_df)
    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # plt.figure(figsize=(12, 5))
    plt.title('Raw signal')
    # plt.xlim(8000, 8080)
    plt.xlabel('Sampling point')
    plt.ylabel('Amplitude')
    plt.subplot(2, 1, 2)
    plt.plot(recoeffs)  # 显示去噪结果
    plt.title('Filtered signal')
    plt.xlabel('Sampling point')
    plt.ylabel('Amplitude')
    plt.subplots_adjust(wspace=0,hspace=0.5)
    plt.suptitle('DWT Threshold Filter', fontsize=16)
    if output_image == 0:
        file_name1 = "%s.png" % "DWT"
        path1 = os.path.join(save_path, file_name1)
        plt.savefig(path1)
    elif output_image == 1:
        file_name1 = "%s.png" % "DWT"
        file_name2 = "%s.jpg" % "DWT"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 2:
        file_name1 = "%s.png" % "DWT"
        file_name2 = "%s.svg" % "DWT"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 3:
        file_name1 = "%s.png" % "DWT"
        file_name2 = "%s.pdf" % "DWT"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    # plt.show()
    plt.close()


    return recoeffs

if __name__=="__main__":
    data = writein('2_GAN_newdata_phy_1000.mat',1)
    # data = writein('image_transformation_DA_newdata.mat')
    print(data[0].shape)
    data_all=DWT_threshold_filter(data[0])
    print(data_all.shape)
