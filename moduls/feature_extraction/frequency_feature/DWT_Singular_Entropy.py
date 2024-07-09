import numpy as np
import pywt
from moduls.feature_extraction.frequency_feature.load import data_load

from utils.table_setting import *
from utils.save_data import save_data_func
from flask import abort

def wrcoef(s, coef_type,wname, level):
    '''

    :param s: input data
    :param coef_type: 'a' or 'd'
    :param wname: name of wavelet
    :param level: The number of layers of wavelet transform
    :return:
    '''
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

def DWT_Singular_Entropy_main(signal,name,N):
    '''

    :param signal: time domain signal
    :param name: Select wavelet function
    :param N: The number of layers of wavelet transform
    :return: WSE; Singular entropy of the signal
    '''

    if N<1:
        abort(400,"The level must be a positive integer.")
    # data = signal.T
    data=signal
    # Wavelet Transform
    # w = pywt.Wavelet(name)
    # coeffs = pywt.wavedec(data, w, level=N)
        # Calculate the singular values of the signal for each frequency bin
    lamda1 = []
    for i in range(1, N + 1):
        D = wrcoef(data, 'd', name, level=i)
        svd=np.linalg.svd(D, compute_uv=False)
        lamda1.append(svd)

    A = wrcoef(data,'a', name, level=N)
    lamda2 = np.linalg.svd(A,compute_uv=False)

    lamda1.append(lamda2)
    # lamda = [lamda2,lamda1]
    lamda=lamda1

    lamda_total=0
    for i in range(len(lamda)):
        lamda_total=np.sum(lamda[i])
    # Calculate the singular entropy of the signal
    L = len(lamda)
    wse = 0
    for i in range(L):
        p= lamda[i] / lamda_total
        P = -(p * np.log(p))
        wse=np.sum(P)

    return wse


if __name__=="__main__":
    pass
    # signal1 = writein('2_GAN_newdata_phy_1000.mat',1)
    # signal = signal1[0]
    # a=DWT_Singular_Entropy_main(signal, 'db1', 3)
    # print(a)
