import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import os
import math
import scipy.io as sio
import pandas as pd
from moduls.data_denoise.EMD.writein import writein
from moduls.data_denoise.physics_based_PSD_filter_threshold.physics_based_PSD_filter_threshold import \
    physics_based_PSD_filter_threshold, signal_denoise

from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func


def PSD_filter(signal_data,fr=37.5, n_ball=8, d_ball=7.92, d_pitch=34.55, alpha=0,frequency_band_max=2000, factor=0.5, sideband_switch=0, sampling_frequency=25600, cut_off_frequency=1000, filter_method=1, filter_order=6,
                   save_path='./', output_file=0, output_image=0):
    '''
    :param signal_data: the signal to denoise,一维数组，例如(10001,)
    :param fr:rotation frequency
    :param n_ball:number of balls
    :param d_ball:diameter of balls
    :param d_pitch:pitch diameter
    :param alpha:initial contact angle
    :param frequency_band_max:the maximum frequency you are interested in, like 2000Hz or 1000Hz;
    :param factor:a scalable factor for the user to tune the algorithm;
    :param sideband_switch: a parameter to define what sideband should be considered,
            0: no sideband considered, only BPFO, BPFI, BSF, FTF;
            1: no sideband considered, only BPFO, BPFI, BSF, FTF, fr
            2: only the BPFI's sideband(±fs) considered;
            3: both BPFI's(±fs) and BSF's(±FTF) sideband considered;
    :param sampling_frequency:the signal sampling frequency, Hz;
    :param cut_off_frequency:the cut_off frequency for the low-pass filter to reduce the high-frequency noise, like 1000Hz;
    :param filter_method: 0: without low-pass filter; 1: with low-pass filter;
    :param filter_order:the defined order of the low-pass filter, integers, like 6, 7, 8;
    :param save_path:path to save
    :param output_file: type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :param output_image: type to save image,0:png,1:jpg,2:svg,3:pdf
    :return:
    '''
    if factor<=0 or factor>1:
        abort(400,"ERROR: The factor must be in (0,1].")
    if cut_off_frequency>sampling_frequency:
        abort(400,"ERROR: The cut off frequency must be less than or equal to the sampling frequency.")
    if frequency_band_max>sampling_frequency:
        abort(400, "ERROR: The frequency band max must be less than or equal to the sampling frequency.")
    if filter_order<1:
        abort(400,"ERROR: The order must be a positive integer.")

    peak_number = physics_based_PSD_filter_threshold(fr, n_ball, d_ball, d_pitch, alpha,frequency_band_max, factor, sideband_switch)
    signal_filtered = signal_denoise(signal_data, sampling_frequency, cut_off_frequency, filter_method, peak_number, filter_order)

    # 保存数据
    save_data_func(data=signal_filtered, output_file=output_file, save_path=save_path,
                   file_name="physics_based_PSD",
                   index_label="Filtered signal")

    # 绘制时域的降噪前后对比图
    plt.subplot(2, 1, 1)
    plt.plot(signal_data)
    plt.title('Raw signal')
    plt.xlabel('Sampling points')
    plt.ylabel("Amplitude")
    plt.subplot(2, 1, 2)
    plt.title('Filtered signal')
    plt.xlabel('Sampling points')
    plt.ylabel("Amplitude")
    plt.plot(signal_filtered)  # 显示去噪结果
    plt.suptitle('Physics-based PSD threshold filter', fontsize=16)
    plt.subplots_adjust(wspace=0,hspace=0.5)
    if output_image == 0:
        file_name1 = "%s.png" % "PSD_filtered_data"
        path1 = os.path.join(save_path, file_name1)
        plt.savefig(path1)
    elif output_image == 1:
        file_name1 = "%s.png" % "PSD_filtered_data"
        file_name2 = "%s.jpg" % "PSD_filtered_data"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 2:
        file_name1 = "%s.png" % "PSD_filtered_data"
        file_name2 = "%s.svg" % "PSD_filtered_data"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 3:
        file_name1 = "%s.png" % "PSD_filtered_data"
        file_name2 = "%s.pdf" % "PSD_filtered_data"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    plt.show()
    plt.close()

    return signal_filtered

if __name__=="__main__":

    # signal1 = writein('2_GAN_newdata_phy_1000.mat',1)
    signal1=writein("1.mat",1)
    # signal1= writein('image_transformation_DA_newdata1.mat')
    signal_data = signal1[0]
    print(signal_data.shape)
    signal_filtered=PSD_filter( signal_data,output_file=1,sampling_frequency=23500)