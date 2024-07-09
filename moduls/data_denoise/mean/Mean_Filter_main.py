import matplotlib.pyplot as plt
import os
import scipy.io as sio
import pandas as pd
import numpy as np
from moduls.data_denoise.EMD.writein import writein

from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func

'''
均值滤波降噪：
    函数ava_filter用于单次计算给定窗口长度的均值滤波
    函数mean_denoise用于指定次数调用ava_filter函数，进行降噪处理
'''


def mean_denoise_main(data, filt_length=10, save_path='./', output_file=0, output_image=0):
    '''
    :param data: innput data,输入为一维数组，例如(10001,)
    :param n: the number of using ava_filter
    :param filt_length: the length of filt
    :param save_path: path to save
    :param output_file: type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :param output_image: type to save image,0:png,1:jpg,2:svg,3:pdf

    '''
    if filt_length <= 0 or filt_length > len(data):
        abort(400, "ERROR: The sliding window length muss be in (0, %d]." % len(data))
    res = np.convolve(data, np.ones((filt_length,)) / filt_length, mode="same")

    # 保存数据
    save_data_func(data=res, output_file=output_file, save_path=save_path,
                   file_name="Mean_filter",
                   index_label="Filtered signal")

    plt.subplot(2, 1, 1)
    plt.plot(data)
    plt.xlabel('Sampling points')
    plt.ylabel('Amplitude')
    plt.title('Raw signal')
    plt.subplot(2, 1, 2)
    plt.plot(res)  # 显示去噪结果
    plt.title('Filtered signal')
    plt.subplots_adjust(wspace=0, hspace=0.5)
    plt.xlabel('Sampling points')
    plt.ylabel('Amplitude')
    plt.suptitle('Mean filter', fontsize=16)

    if output_image == 0:
        file_name1 = "%s.png" % "mean_filter_data"
        path1 = os.path.join(save_path, file_name1)
        plt.savefig(path1)
    elif output_image == 1:
        file_name1 = "%s.png" % "mean_filter_data"
        file_name2 = "%s.jpg" % "mean_filter_data"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 2:
        file_name1 = "%s.png" % "mean_filter_data"
        file_name2 = "%s.svg" % "mean_filter_data"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 3:
        file_name1 = "%s.png" % "mean_filter_data"
        file_name2 = "%s.pdf" % "mean_filter_data"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)

    # plt.show()
    plt.close()
    return res


if __name__ == "__main__":
    data1 = writein("1.mat", 1)
    data0 = data1[0, :1000]

    a = mean_denoise_main(data0)
