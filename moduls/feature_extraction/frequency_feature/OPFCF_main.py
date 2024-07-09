# from feature_extraction.frequency_feature.OPFCF import  DataProcessing,data_load
from moduls.feature_extraction.frequency_feature.OPFCF_final import DataProcessing
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus']=False     # 正常显示负号
import numpy as np
import math
import os
import scipy.io as sio
import pandas as pd
from moduls.feature_extraction.frequency_feature.writein import writein
import logging

from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func

def fault_fre(fr, n_ball, d_ball, d_pitch, alpha,fault_type_list):
    """
    Calculate the theoretical fault characteristic frequency
    :return: Current 4 theoretical fault characteristic frequencies
    """
    f_fault = []
    m = d_ball / d_pitch * math.cos(alpha)
    if fault_type_list[0] == 1:  # BPFO
        f_fault0 = (1 - m) * n_ball * fr / 2
        f_fault.append(f_fault0)
    if fault_type_list[1] == 1:  # BPFI
        f_fault1 = (1 + m) * n_ball * fr / 2
        f_fault.append(f_fault1)
    if fault_type_list[2] == 1:  # BSF
        f_fault2 = (1 - m * m) * d_pitch * fr / (2 * d_ball)
        f_fault.append(f_fault2)
    if fault_type_list[3] == 1:  # FTF
        f_fault3 = (1 - m) * fr / 2
        f_fault.append(f_fault3)
    return f_fault


def OPFCF_main(signal, fault_type_list,fr, order,  fs, switch, delta_f0, threshold, k, n_ball, d_ball, d_pitch, alpha,save_path,output_file=0,output_image=0,RUL_image=False):
    '''

    :param signal: input data,二维数组
    :param fr: rotation frequency
    :param ord: the order of the fault frequency
    :param fs: sampling frequency
    :param fault_type: type of fault, int, 0-FTF, 1-BPFO, 2-BPFI, 3-BSF
    :param delta_f0: fixed frequency interval
    :param return:输出为一维数组
    '''
    if threshold<=1:
        abort(400,'The threshold of variance muss be greater than 1.')
    if k <=0 or k >=100:
        abort(400,'The percent muss be in (0, 100).')
    #计算信号的频率范围，以此判断max order范围
    from scipy.fftpack import fft
    fft_y = fft(signal[0])  # 快速傅里叶变换
    num = len(signal[0])
    x = np.arange(num)  # 频率个数
    half_x = x[range(int(num / 2))]  # 取一半区间
    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
    angle_y = np.angle(fft_y)  # 取复数的角度
    normalization_y = abs_y / num  # 归一化处理（双边频谱）
    normalization_half_y = normalization_y[range(int(num / 2))]  # 由于对称性，只取一半区间（单边频谱）
    findex = np.argwhere(normalization_half_y)
    fmin, fmax = min(findex),max(findex)

    f_fault=fault_fre(fr, n_ball, d_ball, d_pitch, alpha,fault_type_list)
    f_fault_max=float(max(f_fault))
    max_order=int(fmax/f_fault_max)
    if order>max_order or order<1:
        abort(400, 'The order must be in [1, %d].'% max_order)

    data1 = DataProcessing(signal[0],fault_type_list, fr, order, fs, switch, delta_f0, threshold, k, n_ball, d_ball, d_pitch, alpha)
    g_var = data1.var_good()
    OP_FCF_list=[]
    row=signal.shape[0]

    for i in range(row):
        data2 = DataProcessing(signal[i,:],fault_type_list, fr, order, fs,switch, delta_f0, threshold, k, n_ball, d_ball, d_pitch, alpha)
        OP_FCF = data2.probability(g_var)
        OP_FCF_list.append(OP_FCF)
    OP_FCF_ALL=np.array(OP_FCF_list)
    W=3
    OP_FCF_filter_ALL = np.convolve(OP_FCF_ALL, np.ones((W,)) / W, mode='valid')

    # 保存数据
    save_data_func(data=OP_FCF_ALL, output_file=output_file, save_path=save_path,
                   file_name="OPFCF",
                   index_label="OPFCF")

    if RUL_image:
        #绘制OPFCF-RUL趋势图
        plt.figure(figsize=(12, 6), dpi=100)
        plt.plot(np.arange(len(OP_FCF_ALL)), OP_FCF_ALL, color='green', label='Original OPFCF')
        plt.plot(np.arange(len(OP_FCF_filter_ALL)), OP_FCF_filter_ALL, color='red', label='Filtered OPFCF')
        plt.legend()
        plt.title("OPFCF Trend", fontsize=18)
        plt.xlabel("RUL [%]", fontsize=15)
        plt.ylabel("Probability", fontsize=15)
        x_label = np.linspace(0, len(OP_FCF_ALL), 101)
        x_label_use = list(range(101))
        x_label_use.reverse()
        plt.xticks(x_label[::5], x_label_use[::5])
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.grid()

        if output_image == 0:
            file_name1 = "%s.png" % "RUL_OPFCF"
            path1 = os.path.join(save_path, file_name1)
            plt.savefig(path1)
        elif output_image == 1:
            file_name1 = "%s.png" % "RUL_OPFCF"
            file_name2 = "%s.jpg" % "RUL_OPFCF"
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        elif output_image == 2:
            file_name1 = "%s.png" % "RUL_OPFCF"
            file_name2 = "%s.svg" % "RUL_OPFCF"
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        elif output_image == 3:
            file_name1 = "%s.png" % "RUL_OPFCF"
            file_name2 = "%s.pdf" % "RUL_OPFCF"
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)

        # plt.show()
        plt.close()


    return OP_FCF_ALL
if __name__=="__main__":
    signal=writein('1.mat',1)
    OP_FCF_list = OPFCF_main(signal, [0, 1, 1, 1], 35, 9, 25600, 0,10, 3, 3, 8, 7.92, 34.55, 0, './',
                             output_file=0, output_image=0, RUL_image=True)

