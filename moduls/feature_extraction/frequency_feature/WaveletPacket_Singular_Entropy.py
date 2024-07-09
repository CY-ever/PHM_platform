import pywt
import numpy as np
import matplotlib.pyplot as plt
from moduls.feature_extraction.frequency_feature.load import data_load

from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func


def WaveletPacket_Singular_Entropy(signal,name,N):
    '''

    :param signal: input data
    :param name: Select wavelet function
    :param N: The number of layers of wavelet transform

    '''

    if N<1:
        abort(400,"The level must be a positive integer.")

    wp = pywt.WaveletPacket(data=signal, wavelet=name, mode='symmetric', maxlevel=N)
    wpoon = [n.path for n in wp.get_level(N, 'natural')]  #
    SE = np.zeros(len(wpoon))
    for i in range(len(wpoon)):
        m = wp[wpoon[i]].reconstruct(update=False)
        l = [m]
        SE[i] = np.linalg.svd(l, compute_uv=False)

    SE_all = np.sum(SE)
    p = np.zeros(len(SE))
    Singular_Entropy = np.zeros(len(SE))
    for i in range(len(SE)):
        p[i] = SE[i] / SE_all
        Singular_Entropy[i] = -(p[i] * np.log(p[i]))
    Singular_Entropy_sum = np.sum(Singular_Entropy)

    return Singular_Entropy_sum

if __name__ == '__main__':
    pass
    # signal1 = writein('2_GAN_newdata_phy_1000.mat',1)
    # signal = signal1[0]
    # SEP =  WaveletPacket_Singular_Entropy(signal,'db2',3)
    # print(SEP)
    # plt.bar(range(len(SE)),SE)
    # plt.xlabel('number')
    # plt.ylabel('Singular_Entropy')
    # plt.savefig("%s.png" % "Singular_Entropy")
    # plt.show()
