import pandas as pd
import os
import math
import numpy as np
from scipy.fftpack import fft
from scipy.signal import hilbert, detrend, convolve2d, firwin
import matplotlib.pyplot as plt

from moduls.data_denoise.fast_kurtogram.Fast_kurtogram_main import Fast_Kurtogram

from utils.table_setting import *
from utils.save_data import save_data_func
from flask import abort


def fault_fre(fr, n_ball, d_ball, d_pitch, alpha):
    """
    Calculate the theoretical fault characteristic frequency
    :return: Current 4 theoretical fault characteristic frequencies
    """

    m = d_ball / d_pitch * math.cos(alpha)

    # BPFO 外圈理论故障频率
    BPFO = (1 - m) * n_ball * fr / 2

    # BPFI 内圈理论故障频率
    BPFI = (1 + m) * n_ball * fr / 2

    return BPFO, BPFI


def envelope_spectrum(A, fs):
    data = A
    m = len(data)
    f = np.arange(0, m) * fs / m
    f_half = f[0:int(np.round(m / 2))]
    frequency_x = f_half
    H = np.abs(hilbert(data)-np.mean(data))
    HP = np.abs(fft(H-np.mean(H)))*2/m
    HP_half = HP[0:int(np.round(m / 2))]
    envelope_spectrum_y = HP_half
    # plt.figure()
    # plt.plot(f_half,HP_half)
    # plt.xlim([0,800])
    # plt.title('inner')
    # plt.grid(linestyle = '--')
    # plt.show()
    return frequency_x, envelope_spectrum_y

#主函数
def FCF_ratio_method(x, nlevel, order, Fs, fr, n_ball, d_ball, d_pitch, alpha):
    if nlevel<1:
        abort(400,'The level must be a positive integer.')
    if order<1:
        abort(400,'The order must be a positive integer.')
    fc, bw = Fast_Kurtogram(x, nlevel, Fs)
    Wn1 = max([fc - bw / 2, 0.0])
    Wn2 = min([fc + bw / 2, 0.999 * Fs / 2])
    t = np.arange(0, 1, 1/len(x))
    f0 = (Wn1 + Wn2) / 2
    x = x - np.mean(x, 0)
    x0 = x * np.exp(-1j * 2 * np.pi * f0 * t)
    w = firwin(order+1, (Wn2 - Wn1) / 2 / (Fs/2))
    xAn = np.convolve(x0, w, mode='same')

    xEnv = np.abs(2 * xAn)
    xEnv = xEnv - np.mean(xEnv, 0)
    Spec = 1 / len(xEnv) * abs(fft(xEnv))

    fSpec = np.arange(0, len(xEnv)) / len(xEnv) * Fs
    xSpec = Spec[0:int(len(Spec) / 2) + 1]
    fSpec = fSpec[0:int(len(fSpec) / 2) + 1]
    xSpec[1:len(xSpec)] = 2 * xSpec[1:len(xSpec)]
    envelope_spectrum_y = xSpec
    frequency_x = fSpec
    deltaf = frequency_x[1] - frequency_x[0]
    BPFO, BPFI=fault_fre(fr, n_ball, d_ball, d_pitch, alpha)
    BPFIAmplitude = max(envelope_spectrum_y[(frequency_x > (BPFI - 5 * deltaf)) & (frequency_x < (BPFI + 5 * deltaf))])
    BPFOAmplitude = max(envelope_spectrum_y[(frequency_x > (BPFO - 5 * deltaf)) & (frequency_x < (BPFO + 5 * deltaf))])
    FCF_ratio=np.log(BPFIAmplitude/BPFOAmplitude)

    return FCF_ratio

def FCF_ratio_plot(dataset,labels, nlevel, order, fs, fr, n_ball, d_ball, d_pitch, alpha,save_path,output_image):
    '''

    :param dataset: mxn,m是样本组数，例如24x12000
    :return:
    '''
    row=dataset.shape[0]

    FCF_ratio_all=[]
    for i in range(row):
        x1 = dataset[i, :].T
        x = np.squeeze(x1)#变成一维
        FCF_ratio = FCF_ratio_method(x, nlevel, order, fs, fr, n_ball, d_ball, d_pitch, alpha)
        FCF_ratio_all.append(FCF_ratio)
    # # 绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    labels = np.squeeze(labels)
    # list=labels.tolist()
    #

    num_label=np.unique(labels)
    print(num_label)
    if len(num_label) > 4:
        abort(400, "There are up to 4 types of bearing failures.")
    else:
        for i in range(len(labels)):
            if labels[i] == 0:
                plt.hist(FCF_ratio_all[i], bins=6, density=True, facecolor="blue", edgecolor="black", alpha=0.7)
            if labels[i] == 1:
                plt.hist(FCF_ratio_all[i], bins=6, density=True, facecolor="green", edgecolor="black", alpha=0.7)
            if labels[i] == 2:
                plt.hist(FCF_ratio_all[i], bins=6, density=True, facecolor="red", edgecolor="black", alpha=0.7)
            if labels[i] == 3:
                plt.hist(FCF_ratio_all[i], bins=6, density=True, facecolor="yellow", edgecolor="black", alpha=0.7)

    # 显示横轴标签
    # plt.xlabel('log(BPFIAmplitude/BPFOAmplitude)')
    plt.xlabel('log(BPFIAmplitude * BPFOAmplitude ^ (-1))')
    # 显示纵轴标签
    plt.ylabel("Numbers")
    # 显示图标题
    plt.title("FCF-ratio Distribution Histogram")
    #
    if output_image == 0:
        file_name1 = "%s.png" % "FCF_ratio"
        path1 = os.path.join(save_path, file_name1)
        plt.savefig(path1)
    elif output_image == 1:
        file_name1 = "%s.png" % "FCF_ratio"
        file_name2 = "%s.jpg" % "FCF_ratio"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 2:
        file_name1 = "%s.png" % "FCF_ratio"
        file_name2 = "%s.svg" % "FCF_ratio"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 3:
        file_name1 = "%s.png" % "FCF_ratio"
        file_name2 = "%s.pdf" % "FCF_ratio"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    plt.show()
    plt.close()
    return FCF_ratio_all



if __name__ == "__main__":
    #对西安交大第二个工况数据进行处理
    FCF_ratio_outer=[]
    FCF_ratio_inner = []
    FCF_ratio_normal = []
    nlevel = 4  # 分解层数
    Fs = 25600
    fr = 37.5
    n_ball = 8
    d_ball = 7.92
    d_pitch = 34.55
    alpha = 0
    # dataset=writein("3_segmentation_data_100_3000.mat",1)
    # labels=writein("3_segmentation_label_100_3000.mat",1)
    # FCF_ratio_plot(dataset, labels, nlevel, order=3, fs=Fs, fr=fr, n_ball=8, d_