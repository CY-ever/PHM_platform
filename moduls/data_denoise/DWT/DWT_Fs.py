import pywt
from scipy import stats
from moduls.data_denoise.DWT.load import data_load
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import pandas as pd
from moduls.data_denoise.DWT.writein import writein
from flask import abort

from utils.table_setting import *
from utils.save_data import save_data_func


def DWT_Fs_function(signal,name='db1', F=10000, Fs=(3000, 4500), save_path='./',output_file=0,output_image=0):
    '''
    :param save_path: the path to save
    :param signal: time domain signal
    :param name:  Select wavelet mother function
    :param F: the sampling frequency of the signal
    :param Fs: filter noise in this frequency band
    :param output_file: type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :param output_image: type to save image,0:png,1:jpg,2:svg,3:pdf
    :return: s_rec; Signal after filtering noise
    '''
    if Fs[0]<0:
        abort(400, "ERROR: The minimum value of the frequency must be a positive number.")
    elif Fs[0] > Fs[1]:
        abort(400,"ERROR: The maximum value of the frequency must be greater than the minimum value.")
    elif Fs[1] > F:
        abort(400, "ERROR: The maximum value of the frequency must be less than or equal to the sampling frequency.")
    # % Determine which region of the target frequency after wavelet transform is after low-pass filtering
    elif Fs[1] <= F / 2:
        fs = F / 2
        LISTfs = [fs]
        while fs > 5:
            fs = fs / 2
            LISTfs.append(fs)
        n2=n1=0
        L = len(LISTfs)
        for i in range(L - 1):
            if LISTfs[i + 1] <= Fs[0] | Fs[0] < LISTfs[i]:
                n1 = i + 1

        for i in range(L - 1):
            if LISTfs[i + 1] < Fs[1] | Fs[1] <= LISTfs[i]:
                n2 = i + 1

    # % wavelet transform
    # % Extraction Extraction Coefficients
        data = signal.T.tolist()  # 将np.ndarray()转为列表
        w = pywt.Wavelet(name)

        coeffs = pywt.wavedec(data, w, level=n1)

        A = coeffs[0]
        D = coeffs[1:]
        # % Assign the target frequency coefficients to 0
        for n in range(n2, n1):
            D[n][:, 0] = 0

        A = A.T
        DD = []
        for n in range(n1):
            Dd = D[n].T
            DD.append(Dd)
        DD.append(A)
        Dz = DD
        # % Reconstruct the filtered wavelet domain signal

        s_rec = pywt.waverec(Dz, name)  # 信号重构


    elif Fs[1] > F/ 2:
    # % The target frequency band is in the high frequency region
        if Fs[0] >= F / 2:
            # % wavelet transform
            # % Extraction Extraction Coefficients
            coeffs = pywt.wavedec(signal, name, level=1)
            A = coeffs[0]
            D = coeffs[1:]
            # % Assign the target frequency coefficients to 0
            D = D * 0
            D.T.append(A.T)
            Dz = D.T
            # % Reconstruct the filtered wavelet domain signal
            s_rec = pywt.waverec(Dz, name)

        # % The target high frequency is in the high frequency area, and the target low frequency is in the low frequency area
        elif Fs[0] < F / 2:
            fs = F / 2
            LISTfs = [fs]
            while fs > 5:
                fs = fs / 2
                LISTfs.append(fs)
            n1=0
            L = len(LISTfs)
            for i in range(L - 1):
                if LISTfs[i + 1] <= Fs[0] | Fs[0] < LISTfs[i]:
                    n1 = i + 1

            # % wavelet transform
            # % Extraction Extraction Coefficient
            coeffs = pywt.wavedec(signal, name, level=n1)
            # print(type(coeffs))
            A = coeffs[0]
            D = coeffs[1:]
            # % Assign the target frequency coefficients to 0
            for n in range(n1):
                D[n][:, 0] = 0

            A = A.T
            DD = []
            for n in range(n1):
                Dd = D[n].T
                DD.append(Dd)
            DD.append(A)
            Dz=DD

            s_rec = pywt.waverec(Dz, name)
    # 保存数据
    save_data_func(data=s_rec, output_file=output_file, save_path=save_path,
                   file_name="DWT_frequency_band_filter",
                   index_label="Filtered signal")

    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.subplots_adjust(wspace=0, hspace=0.5)  # 调整子图间距
    plt.title('Raw signal')
    plt.xlabel('Sampling points')
    plt.ylabel('Amplitude')
    plt.subplot(2, 1, 2)
    plt.title('Filtered signal')
    plt.xlabel('Sampling points')
    plt.ylabel('Amplitude')

    plt.plot(s_rec)  # 显示去噪结果
    plt.suptitle('DWT Frequency Band Filter',fontsize=16)
    if output_image==0:
        file_name1 = "%s.png" % "DWT"
        path1 = os.path.join(save_path, file_name1)
        plt.savefig(path1)
    elif output_image==1:
        file_name1 = "%s.png" % "DWT"
        file_name2 = "%s.jpg" % "DWT"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image==2:
        file_name1 = "%s.png" % "DWT"
        file_name2 = "%s.svg" % "DWT"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image==3:
        file_name1 = "%s.png" % "DWT"
        file_name2 = "%s.pdf" % "DWT"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    # plt.show()
    plt.close()
    return s_rec

if __name__ == "__main__":
    data = writein('2_GAN_newdata_phy_1000.mat', 1)
    signal = data[0]
    new_wp = DWT_Fs_function(signal)
    print(new_wp.shape)

