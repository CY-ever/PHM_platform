# Date: 2021.3.23
# Version: 0.2.0
# @author: Yuanheng Mu & Runkai He
# Description: Functions to filter signal with EMD (empirical mode decomposition)

from PyEMD import EMD
from scipy.signal import argrelextrema
import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt
from moduls.data_denoise.EMD.writein import writein
import os
import scipy.io as sio
import pandas as pd
import logging
from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func



def filter_with_emd_main(raw_data,N=3,save_path='./',output_file=0,output_image=0):
    '''

    :param raw_data:
    :param N:die maximale Anzahl von IMFs zu extrahieren
    :return:
    '''
    emd = EMD()
    IMFs = emd(raw_data)
    IMFs_max=IMFs.shape[0]-1
    if N>=IMFs.shape[0] or N<=0:
        abort(400,'ERROR: The number of IMFs selected muss be in [1, %d].'% (IMFs_max-1))
    filtered_data=np.sum(IMFs[N:], axis=0)
    # 绘制IMFs图
    plt.figure(figsize=(15,12))
    n = len(IMFs) + 1-N
    # plt.figure(figsize=(20, 18))
    plt.subplot(n, 1, 1)
    plt.plot(raw_data)
    # plt.subplots_adjust(wspace=0, hspace=1.5)
    plt.title("EMD decomposition",fontsize=18)
    plt.ylabel('Original',fontsize=15)

    for i in range(len(IMFs)-N):
        plt.subplot(n, 1, i + 2)
        plt.plot(IMFs[i+N])

        if (i == len(IMFs) - 1-N):
            plt.ylabel("Residual",fontsize=15)
        else:
            plt.ylabel("IMF" + str(i + 1),fontsize=15)
    plt.subplots_adjust(wspace=0,hspace=0.5)
    plt.xlabel('Sampling points',fontsize=15)
    plt.tight_layout()
    if output_image == 0:
        file_name1 = "%s.png" % "EMD_IMFs"
        path1 = os.path.join(save_path, file_name1)
        plt.savefig(path1)
    elif output_image == 1:
        file_name1 = "%s.png" % "EMD_IMFs"
        file_name2 = "%s.jpg" % "EMD_IMFs"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 2:
        file_name1 = "%s.png" % "EMD_IMFs"
        file_name2 = "%s.svg" % "EMD_IMFs"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 3:
        file_name1= "%s.png" % "EMD_IMFs"
        file_name2 = "%s.pdf" % "EMD_IMFs"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    # plt.show()
    plt.close()
    #绘制降噪图
    plt.subplot(2, 1, 1)
    plt.plot(raw_data)
    plt.gcf().subplots_adjust(left=0.15)
    plt.ylabel('Amplitude')
    plt.xlabel('Sampling points')
    plt.title("Raw signal")

    plt.subplot(2, 1, 2)
    plt.subplots_adjust(wspace=0, hspace=0.5)
    plt.gcf().subplots_adjust(left=0.15)
    plt.ylabel('Amplitude')
    plt.xlabel('Sampling points')
    plt.title("Filtered signal")
    plt.suptitle('EMD ', fontsize=16)
    plt.plot(filtered_data)

    if output_image == 0:
        file_name1 = "%s.png" % "EMD_filtered_data"
        path1 = os.path.join(save_path, file_name1)
        plt.savefig(path1)
    elif output_image == 1:
        file_name1 = "%s.png" % "EMD_filtered_data"
        file_name2 = "%s.jpg" % "EMD_filtered_data"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 2:
        file_name1 = "%s.png" % "EMD_filtered_data"
        file_name2 = "%s.svg" % "EMD_filtered_data"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 3:
        file_name1 = "%s.png" % "EMD_filtered_data"
        file_name2 = "%s.pdf" % "EMD_filtered_data"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    # plt.show()
    plt.close()

    # 保存数据
    save_data_func(data=filtered_data, output_file=output_file, save_path=save_path,
                   file_name="EMD_filter",
                   index_label="Filtered signal")

    return filtered_data

if __name__=="__main__":

    # c1 = writein('2_GAN_newdata_phy_1000.mat',1)
    c1 = writein('1.mat', 1)
    c=c1[0]
    # print(c1.shape)
    # print(c.shape)
    b= filter_with_emd_main(c,N=9)