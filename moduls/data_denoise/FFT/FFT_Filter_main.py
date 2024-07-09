import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from moduls.data_denoise.EMD.writein import writein
import os
import scipy.io as sio
from scipy import signal
from scipy.signal import detrend
import pandas as pd

from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func



def FFT_filter(raw_data,critical_freqs=(0, 2000), mode='lowpass',save_path='./',output_file=0,output_image=0):
    '''

    :param raw_data: input time domain signal,输入为一维数组，例如(10001,)
    :param critical_freqs: limited frequency
    :param mode: string, 'lowpass','bandpass','highpass'
    :param save_path: path to save
    :param output_file: type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :param output_image: type to save image,0:png,1:jpg,2:svg,3:pdf
    :return:
    '''
    if critical_freqs[0]<0 or critical_freqs[1]<0:
        abort(400,"ERROR: The frequency value must be greater than 0.")

    fft_y = fft(raw_data)  # 快速傅里叶变换

    N = len(raw_data)
    x = np.arange(N)  # 频率个数
    half_x = x[range(int(N / 2))]  # 取一半区间
    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
    normalization_y = abs_y / N  # 归一化处理（双边频谱）
    normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）
    filtered_data_fft=fft_y.copy()

    # 滤波
    if mode == "lowpass": #低频 保留(0, Max)
        noised_indices = np.where(half_x >= critical_freqs[1])
        filtered_data_fft[noised_indices] = 0
    elif mode=="bandpass": #频带 保留[Min, Max]频率范围
        if critical_freqs[0]>=critical_freqs[1]:
            abort(400,"ERROR: The maximum value of the frequency must be greater than the minimum value.")
        else:
            noised_indices1 = np.where(half_x > critical_freqs[1])
            filtered_data_fft[noised_indices1] = 0
            noised_indices2= np.where(half_x< critical_freqs[0])
            filtered_data_fft[noised_indices2] = 0
    else: #高频 保留（Min, )
        noised_indices = np.where(half_x <= critical_freqs[0] )
        filtered_data_fft[noised_indices] = 0

    abs_filtered_data = np.abs(filtered_data_fft)

    normalization_fft = abs_filtered_data / N
    filtered_data_half = normalization_fft[range(int(N / 2))] #单边频谱


        # 信号降噪前后的频谱图
    plt.subplot(2, 1, 1)
    plt.plot(half_x, normalization_half_y)
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency [Hz]')
    plt.title('Raw signal')
    plt.subplot(2, 1, 2)
    plt.plot(half_x, filtered_data_half)
    plt.title('Filtered signal')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.suptitle('Frequency', fontsize=16)
    plt.subplots_adjust(wspace=0, hspace=0.5)
    # plt.show()
    plt.close()

    # 最后一步就是逆转换了，将频域数据转换回时域数据
    new_f_clean =ifft(filtered_data_fft)
    new_f_clean=np.real(new_f_clean)


    # 保存数据
    save_data_func(data=new_f_clean, output_file=output_file, save_path=save_path,
                   file_name="FFT",
                   index_label="Filtered signal")

    # 绘图 信号降噪前后的时域对比图
    plt.subplot(2, 1, 1)
    plt.plot(raw_data)
    plt.title('Raw signal')
    plt.xlabel('Sampling points')
    # plt.xlim(0,200)
    plt.ylabel('Amplitude')
    plt.subplot(2, 1, 2)
    plt.plot(new_f_clean)
    plt.title('Filtered signal')
    plt.xlabel('Sampling points')
    # plt.xlim(0, 200)
    plt.ylabel('Amplitude')
    plt.suptitle('FFT', fontsize=16)
    plt.subplots_adjust(wspace=0, hspace=0.5)
    if output_image == 0:
        file_name1 = "%s.png" % "FFT"
        path1 = os.path.join(save_path, file_name1)
        plt.savefig(path1)
    elif output_image == 1:
        file_name1 = "%s.png" % "FFT"
        file_name2 = "%s.jpg" % "FFT"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 2:
        file_name1 = "%s.png" % "FFT"
        file_name2 = "%s.svg" % "FFT"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 3:
        file_name1 = "%s.png" % "FFT"
        file_name2 = "%s.pdf" % "FFT"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    # plt.show()
    plt.close()


    return new_f_clean

if __name__ =="__main__":
    a = writein('1.mat',1)
    FFT_filter(a[0,:10000],critical_freqs=(0, 2000),mode="lowpass")
    print(a.shape)