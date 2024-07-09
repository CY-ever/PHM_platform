import pywt
import numpy as np
import matplotlib.pyplot as plt
# from load import data_load
import os
import scipy.io as sio
import pandas as pd
from moduls.data_denoise.WaveletPacket.writein import writein

from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func


def WaveletPacket_Filter(signal, name='db1', N=5, save_path='./', output_file=0, output_image=0):
    '''
    :param signal: input time domain signal
    :param name: name of wavelet mother function
    :param N: number of levels of wavelet packet decomposition
    :param save_path: path to save
    :param output_file: type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :param output_image: type to save image,0:png,1:jpg,2:svg,3:pdf
    :return:
    '''
    if N < 1:
        abort(400, "ERROR: The level must be a positive integer.")
    # t = np.linspace(0, len(signal), len(signal))
    wp = pywt.WaveletPacket(data=signal, wavelet=name, mode='symmetric', maxlevel=N)
    wppon = [node.path for node in wp.get_level(N, 'natural')]
    for i in wppon:
        if i != wppon[0]:
            wp[i] = wp[i].data * 0
    new_wp = wp.reconstruct(update=False)
    # 保存数据
    save_data_func(data=new_wp, output_file=output_file, save_path=save_path,
                   file_name="Wavelet_packet_simple_filter",
                   index_label="Filtered signal")

    # 绘图
    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.title('Raw signal')
    plt.xlabel('Sampling points')
    plt.ylabel('Amplitude')
    plt.subplot(2, 1, 2)
    plt.plot(new_wp)
    plt.title('Filtered signal')
    plt.xlabel('Sampling points')
    plt.ylabel('Amplitude')
    plt.suptitle('Wavelet Packet Simple Filter', fontsize=16)
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


if __name__ == '__main__':
    signal1 = writein('2_GAN_newdata_phy_1000.mat', 1)
    signal = signal1[0]
    WaveletPacket_Filter(signal)
