import pywt
import numpy as np
import matplotlib.pyplot as plt
from moduls.feature_extraction.frequency_feature.load import data_load

from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func


def WaveletPacket_EnergyEntropy(signal,name,N):
    '''

    :param signal: input data
    :param name: Select wavelet function
    :param N: The number of layers of wavelet transform

    '''
    if N<1:
        abort(400,"The level must be a positive integer.")
    wp = pywt.WaveletPacket(data=signal,wavelet= name,mode='symmetric',maxlevel=N)
    wopon = [n.path for n in wp.get_level(N,'natural')] #
    Energy = []
    for i in wopon:
        n = wp[i].reconstruct(update=True)
        Ln = np.linalg.norm(n,ord=2)**2
        Energy.append(Ln)
    Energy_all = np.sum(Energy)
    P = np.zeros(len(Energy))
    Energy_Entropy = np.zeros(len(Energy))
    for i in range(len(Energy)):
        P[i]= Energy[i]/Energy_all
        Energy_Entropy[i] = - (P[i]*np.log(P[i]))
    Energy_Entropy_sum = np.sum(Energy_Entropy)

    return Energy_Entropy_sum


if __name__ == '__main__':
    pass
    # signal1 = writein('2_GAN_newdata_phy_1000.mat',1)
    # signal = signal1[0]
    # EA = WaveletPacket_EnergyEntropy(signal,'sym2',3)
    # print(EA)
    # plt.xlabel('3ceng')
    # plt.ylabel('Energy Entropy Value')
    # plt.bar(range(len(E)),E)
    # plt.show()
