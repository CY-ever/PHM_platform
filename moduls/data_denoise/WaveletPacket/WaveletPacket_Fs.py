import pywt
import numpy as np
import matplotlib.pyplot as plt
from moduls.data_denoise.WaveletPacket.writein import writein
import os
import scipy.io as sio
import pandas as pd
from flask import abort
# from FFT_filter import FFT_filter

from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func

def WaveletPacket_Fs(signal, name='db1', sampling_frequency=10000, Fs_band=(3000, 4500),save_path='./',output_file=0,output_image=0):
    '''
    :param signal: time domain signal,一维数组，例如(10001,)
    :param name: name of wavelet mother function
    :param sampling_frequency:sampling_frequency, like 10000Hz,23600Hz
    :param Fs_band: filter noise in this frequency band
    :param save_path: path to save
    :param output_file: type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :param output_image: type to save image,0:png,1:jpg,2:svg,3:pdf
    :return: s_rec; Signal after filtering noise
    '''
    Fmax=sampling_frequency/2
    fs= sampling_frequency / 4 #这里fs为最大频率除以2，这里相当于采样频率除以4
    LISTfs = [fs]
    while fs > 5:
        fs = fs / 2
        LISTfs.append(fs)

    L = len(LISTfs)
    if Fs_band[0] > Fs_band[1]:
        abort(400, "ERROR: The maximum value of the frequency must be greater than the minimum value.")
        # abort(400,'error interval')
    elif Fs_band[0] < LISTfs[-1]:
        abort(400,"ERROR: The minimum value of the frequency is %d." % (LISTfs[-1]))
        # abort(400, 'interval minimum error')
    elif Fs_band[1] > Fmax:
        abort(400, "ERROR: The maximum value of the frequency is %d." % Fmax)
        # abort(400, 'interval maximum error')
    else:
        # % % wavelet packet transform
        n1=n2=0
        List = list(np.arange(LISTfs[-1], Fmax, LISTfs[-1]))
        l = len(List)
        # % Determine which region the target frequency is in after wavelet transform
        for i in range(l - 1):
            # global n1
            if List[i + 1] > Fs_band[0] | Fs_band[0] >= List[i]:
                n1 = i
                break


        for i in range(l - 1):

            if Fs_band[1] < Fmax:
                if List[i + 1] > Fs_band[1] | Fs_band[1] >= List[i]:
                    n2 = i
                    break


            elif Fs_band[1] == Fmax:
                if List[i + 1] >= Fs_band[1] | Fs_band[1] >= List[i]:
                    n2 = i
                    break

        # % Assign the target frequency coefficients to 0

        wp = pywt.WaveletPacket(data=signal, wavelet=name, mode='symmetric', maxlevel=L)

        nodes_ord = [node.path for node in wp.get_level(L, 'freq')]
        for ii in range(n1,n2):
            wp[nodes_ord[ii]].data = wp[nodes_ord[ii]].data * 0
        # Reconstruct the filtered wavelet domain signal
        new_wp = wp.reconstruct(update=False)

        # 保存数据
        save_data_func(data=new_wp, output_file=output_file, save_path=save_path,
                       file_name="Wavelet_packet_frequency_band_filter",
                       index_label="Filtered signal")

        plt.subplot(2, 1, 1)
        plt.plot( signal)
        plt.title('Raw signal')
        plt.xlabel('Samping points')
        plt.ylabel('Amplitude')
        plt.subplot(2, 1, 2)
        plt.title('Filtered signal')
        plt.xlabel('Samping points')
        plt.ylabel('Amplitude')
        plt.plot( new_wp)  # 显示去噪结果
        plt.suptitle('Wavelet Packet Frequency Band Filter', fontsize=16)
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


if __name__ == "__main__":

    data = writein('2_GAN_newdata_phy_1000.mat',1)
    signal = data[0]
    new_wp = WaveletPacket_Fs(signal)
    # print(new_wp)
