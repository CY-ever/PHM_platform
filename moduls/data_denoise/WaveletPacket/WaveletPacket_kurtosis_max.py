import pywt
from scipy import stats
# from load import data_load
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import pandas as pd
from moduls.data_denoise.WaveletPacket.writein import writein
from flask import abort

from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func

def WaveletPacket_kurtosis_max(signal, name='db1', N=3, k=6,save_path='./',output_file=0,output_image=0):
    '''
    :param signal: input time domain signal,一维数组，例如(10001,)
    :param name: name of wavelet mother function
    :param N: number of levels of wavelet packet decomposition
    :param k: number of selected kurtosis
    :param save_path: path to save
    :param output_file: type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :param output_image: type to save image,0:png,1:jpg,2:svg,3:pdf
    :return:
    '''
    if N < 1:
        abort(400, "ERROR: The level must be a positive integer.")
    if k >= 2** N:
        abort(400, "ERROR: The number of kurtosis selected must be in [1, %d]." % (2**N-1))
        # abort(400,'Error in the number of kurtosis')
    else:
        wp = pywt.WaveletPacket(data=signal, wavelet=name, mode='symmetric', maxlevel=N)
        wpoon = [n.path for n in wp.get_level(N, 'natural')]
        cks = []
        for i in range(len(wpoon)):
            cks1 = stats.kurtosis(wp[wpoon[i]].data, fisher=False)
            cks.append(cks1)
        cks.sort(reverse=True)
        cL=cks

        for i in range(len(wpoon)):
            for j in range(k):
                if cL[j] == stats.kurtosis(wp[wpoon[i]].data, fisher=False):
                    wp[wpoon[i]].data = wp[wpoon[i]].data*0

        new_wp = wp.reconstruct(update=False)

        # 保存数据
        save_data_func(data=new_wp, output_file=output_file, save_path=save_path,
                       file_name="Wavelet_packet_kurtosis_filter",
                       index_label="Filtered signal")

        plt.subplot(2, 1, 1)
        plt.plot(signal)
        plt.xlabel('Sampling points')
        plt.ylabel('Amplitude')
        plt.title('Raw signal')
        plt.subplot(2, 1, 2)
        plt.title('Filtered signal')
        plt.xlabel('Sampling points')
        plt.ylabel('Amplitude')
        plt.plot(new_wp)  # 显示去噪结果
        plt.suptitle('Wavelet Packet Kurtosis Filter', fontsize=16)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        if output_image == 0:
            file_name1 = "%s.png" % "WaveletPacket"
            path1 = os.path.join(save_path, file_name1)
            plt.savefig(path1)
        elif output_image == 1:
            file_name1 = "%s.png" % "WaveletPacket"
            file_name2 = "%s.jpg" % "WaveletPacket"
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        elif output_image == 2:
            file_name1 = "%s.png" % "WaveletPacket"
            file_name2 = "%s.svg" % "WaveletPacket"
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        elif output_image == 3:
            file_name1 = "%s.png" % "WaveletPacket"
            file_name2 = "%s.pdf" % "WaveletPacket"
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        # plt.show()
        plt.close()

    return new_wp
if __name__=="__main__":
    data =  writein('2_GAN_newdata_phy_1000.mat',1)
    signal = data[0]
    new_wp=WaveletPacket_kurtosis_max(signal)
    # print(new_wp)