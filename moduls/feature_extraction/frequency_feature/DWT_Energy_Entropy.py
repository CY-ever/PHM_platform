import numpy as np
import pywt
from moduls.feature_extraction.frequency_feature.load import data_load
from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func

def wrcoef(s, coef_type,wname, level):
    N = len(s)
    w = pywt.Wavelet(wname)
    a = s
    ca = []
    cd = []
    for i in range(level):
        (a, d) = pywt.dwt(a, w, 'symmetric')  # 将a作为输入进行dwt分解
        ca.append(a)
        cd.append(d)
    rec_a = []
    rec_d = []
    if coef_type == 'a':
        for i, coeff in enumerate(ca):
            coeff_list = [coeff, None] + [None] * i
            rec_a.append(pywt.waverec(coeff_list, w)[0:N])  # 进行重构
        return rec_a
    if coef_type == 'd':
        for i, coeff in enumerate(cd):
            coeff_list = [None, coeff] + [None] * i
            rec_d.append(pywt.waverec(coeff_list, w)[0:N])  # 进行重构
        return rec_d

def DWT_Energe_Entropy_main(signal, name, N):
    '''

    :param signal: time domain signal
    :param name: Select wavelet function
    :param N: The number of layers of wavelet transform

    '''

    if N<1:
        abort(400,"The level must be a positive integer.")

    # data = signal.T
    data=signal
    eD = []
    for i in range(1,N+1):
        e = np.linalg.norm(wrcoef(data,'d', name, i), ord=2) ** 2
        eD.append(e)

    eA = np.linalg.norm(wrcoef(data,'a', name, N), ord=2) ** 2
    eD.append(eA)
    E = eD
    E_total = sum(E)
    L = len(E)
    energyE = 0
    for i in range(L):
        p = E[i] / E_total
        # P = -(p * np.log(p))
        energyE = energyE + (p * np.log(p))

    return -energyE

if __name__=="__main__":
    signal1 = writein('2_GAN_newdata_phy_1000.mat',1)
    signal = signal1[0]
    print(signal.shape)
    a=DWT_Energe_Entropy_main(signal, 'db1', 3)
    print(a)