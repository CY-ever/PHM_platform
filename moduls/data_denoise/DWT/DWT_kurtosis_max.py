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

def DWT_kurtosis_max_function(signal, name='db1', N=3, k=3,save_path='./',output_file=0,output_image=0):
    '''

    :param save_path: path to save
    :param signal: input data
    :param name: name of wavelet transfer
    :param N: number of wavelet decompositions
    :param k: number of kurtosis selected
    :param output_file: type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :param output_image: type to save image,0:png,1:jpg,2:svg,3:pdf

    '''

    data = signal.T.tolist()  # 将np.ndarray()转为列表
    w = pywt.Wavelet(name)
    coeffs = pywt.wavedec(data, w, level=N)
    A=coeffs[0]
    D=[coeffs[1:]]
    if k > N or k<1:
        abort(400, "ERROR: The number of kurtosis selected must be in [1, %d]." % N)
        # raise ('Error in the number of kurtosis')
    elif N == 1:
        for x in range(D):
            max = stats.kurtosis(x, fisher=False)
            CK = []
            CK.append(max)
            CK.sort(reverse=True)
            cL=CK
        # % Filter the frequency band signal corresponding to the first k kurtosis values
        for m in range( k):
            if max == cL[m]:
                D = D * 0

        Dz = [A.T, D.T]
        # % Reconstruct the filtered wavelet domain signal
        s_rec=pywt.waverec(Dz, name)  # 信号重构
    elif N > 1:
        # % Wavelet Transform Extraction Extraction Coefficients
        coeffs = pywt.wavedec(data,w, level=N)
        A = coeffs[0]
        D = coeffs[1:]
        print(D[0])
        # % Calculate the kurtosis value of the signal for each frequency band
        cks = []
        for i in range( N):
            cks1 = stats.kurtosis(D[i], fisher=False)
            cks.append(cks1)

        cks4 = stats.kurtosis(A, fisher=False)
        # % Sorting of kurtosis values (large to small)
        cks.append(cks4)
        cA=cks
        cA.sort(reverse=True)
        cL=cA
        # % Filter the frequency band signal corresponding to the first k kurtosis values
        for i in range( N):
            for m in range( k):
                if stats.kurtosis(D[i], fisher=False) == cL[m]:
                    D[i] = D[i] * 0
                elif stats.kurtosis(A, fisher=False) == cL[m]:
                    A = A * 0

        Dz=[]
        Dz.append(A.T)
        for n in range(N):
            Dd = D[n].T
            Dz.append(Dd)
        # % Reconstruct the filtered wavelet domain signal
        s_rec= pywt.waverec(Dz, name)
    else:
        abort(400, "ERROR: The level must be a positive integer.")

    # 保存数据
    save_data_func(data=s_rec, output_file=output_file, save_path=save_path,
                   file_name="DWT_kurtosis_filter",
                   index_label="Filtered signal")

    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.subplots_adjust(wspace=0, hspace=0.5)
    plt.title('Raw signal')
    plt.xlabel('Sampling points')
    plt.ylabel('Amplitude')
    plt.subplot(2, 1, 2)
    plt.title('Filtered signal')
    plt.xlabel('Sampling points')
    plt.ylabel('Amplitude')
    plt.plot(s_rec)  # 显示去噪结果
    plt.suptitle('DWT Kurtosis Filter', fontsize=16)
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

    return s_rec


if __name__=="__main__":
    # name = pywt.Wavelet('sym8')
    # data = data_load('image_transformation_DA_newdata.mat', "data", ".mat", "row")
    # data = writein('image_transformation_DA_newdata.mat')
    data = writein('2_GAN_newdata_phy_1000.mat',1)
    signal = data[0]
    print(signal.shape)
    s_rec=DWT_kurtosis_max_function(signal)
    print(s_rec.shape)
    # print(np.dot(0.5, (102.7938 + 147.7264)))