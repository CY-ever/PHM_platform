import numpy as np
import pywt
from scipy import stats
import os
import scipy.io as sio
import pandas as pd
from flask import abort
from moduls.feature_extraction.frequency_feature.OPFCF_final import DataProcessing
import math
from scipy.fftpack import fft
from scipy.signal import hilbert, detrend, convolve2d, firwin
import matplotlib.pyplot as plt
from moduls.feature_extraction.frequency_feature.writein import writein
from moduls.data_denoise.fast_kurtogram.Fast_kurtogram_main import Fast_Kurtogram

from utils.table_setting import *
from utils.save_data import save_data_func

# 主函数
def ALL_frequency_features(dataset,labels, f_features_list=(1,1,1,1,1,1,1,1,1), DWT_EnergyEntropy_name='db1', DWT_EnergyEntropy_N=3, DWT_SingularEntropy_name='db1',
    DWT_SingularEntropy_N=3, WaveletPacket_EnergyEntropy_name='db1', WaveletPacket_EnergyEntropy_N=5,
    WaveletPacket_SingularEntropy_name='db1', WaveletPacket_SingularEntropy_N=5, OPFCF_fault_type_list= (1, 0, 0, 0),
    OPFCF_fr=35, OPFCF_order=9,  OPFCF_fs=25600, OPFCF_switch=1, OPFCF_delta_f0=10, OPFCF_threshold=3,
    OPFCF_k=3, OPFCF_n_ball=8, OPFCF_d_ball=7.92, OPFCF_d_pitch=34.55, OPFCF_alpha=0, OPFCF_RUL_image=True, data_SBN0=None,
                                          f_normal_features=(0,0,1,1,1,1,1,1,1,1,1,1,1),emd_fr=25,
    emd_n_ball=16, emd_d_ball=22.225, emd_d_pitch=125.26 , emd_alpha=0, emd_fs=10000, emd_fault_type=0, emd_n=3, emd_ord=3, emd_limit=2000,
    FCF_ratio_nlevel=6, FCF_ratio_order=8, FCF_ratio_fs=10000, FCF_ratio_fr=25, FCF_ratio_n_ball=8,
    FCF_ratio_d_ball=7.92, FCF_ratio_d_pitch=34.55, FCF_ratio_alpha=0,FCF_ratio_image=True, save_path='./',output_image=0):

    global g_var, f_name
    if f_features_list[4] == 1:
        if labels is None:
            abort(400, "The label file is missing.")
        else:
            labels_new=np.squeeze(labels)
            sort = np.argsort(-labels_new)
            dataset = dataset[sort]
            labels=labels[sort]
            data1 = DataProcessing(dataset[0], OPFCF_fault_type_list,
        OPFCF_fr, OPFCF_order, OPFCF_fs, OPFCF_switch, OPFCF_delta_f0, OPFCF_threshold,
        OPFCF_k, OPFCF_n_ball, OPFCF_d_ball, OPFCF_d_pitch, OPFCF_alpha)
            g_var = data1.var_good()
    OPFCF_list=[]
    dataset_f_features_all = []
    for i in range(dataset.shape[0]):
        signal=dataset[i]
        f_features_all=[]
        f_name=[]
        if f_features_list[0]==1:
            features=DWT_Energe_Entropy_main(signal, DWT_EnergyEntropy_name, DWT_EnergyEntropy_N)
            f_features_all.append(features)
            f_name.append('DWT_EnergyEntropy')
        if f_features_list[1]==1:
            features=DWT_Singular_Entropy_main(signal, DWT_SingularEntropy_name, DWT_SingularEntropy_N)
            f_features_all.append(features)
            f_name.append('DWT_SingularEntropy')
        if f_features_list[2]==1:
            features=WaveletPacket_EnergyEntropy(signal,WaveletPacket_EnergyEntropy_name,WaveletPacket_EnergyEntropy_N)
            f_features_all.append(features)
            f_name.append('WP_EnergyEntropy')
        if f_features_list[3]==1:
            features=WaveletPacket_Singular_Entropy(signal,WaveletPacket_SingularEntropy_name,WaveletPacket_SingularEntropy_N)
            f_features_all.append(features)
            f_name.append('WP_SingularEntropy')
        if f_features_list[4]==1:
            features=OPFCF_main(signal, g_var,OPFCF_fault_type_list, OPFCF_fr, OPFCF_order, OPFCF_fs, OPFCF_switch, OPFCF_delta_f0, OPFCF_threshold, OPFCF_k, OPFCF_n_ball, OPFCF_d_ball, OPFCF_d_pitch, OPFCF_alpha)
            f_features_all.append(features)
            OPFCF_list.append(features)
            f_name.append('OPFCF')
        if f_features_list[5]==1:
            features=SBN(signal,data_SBN0)
            f_features_all.append(features)
            f_name.append('GMR-FFT')
        if f_features_list[6] == 1:
            features,f_normal_name = frequency_normal_features(signal,f_normal_features)
            f_features_all.extend(features)
            f_name.extend(f_normal_name)
        if f_features_list[7] == 1:
            features = emd_feature(signal,emd_fr, emd_n_ball, emd_d_ball, emd_d_pitch, emd_alpha,emd_fs,emd_fault_type,emd_n,emd_ord,emd_limit)
            f_features_all.append(features)
            f_name.append('Envelope spectrum')
        if f_features_list[8] == 1:
            features = FCF_ratio_method(signal, FCF_ratio_nlevel, FCF_ratio_order, FCF_ratio_fs, FCF_ratio_fr, FCF_ratio_n_ball, FCF_ratio_d_ball, FCF_ratio_d_pitch, FCF_ratio_alpha)
            f_features_all.append(features)
            f_name.append('FCF-ratio')


        dataset_f_features_all.append(f_features_all)
    OPFCF_ALL=np.array(OPFCF_list)
    dataset_f_features_all=np.array(dataset_f_features_all)
    print("OPFCF_image", OPFCF_RUL_image)
    if f_features_list[4]==1 and OPFCF_RUL_image:
        OPFCF_RUL_plot(OPFCF_ALL, save_path, output_image)
    if f_features_list[8] == 1 and FCF_ratio_image:
        if labels is None:
            abort(400, "The label file is missing.")
        else:
            FCF_ratio_plot(dataset, labels, FCF_ratio_nlevel, FCF_ratio_order, FCF_ratio_fs, FCF_ratio_fr, FCF_ratio_n_ball,
                           FCF_ratio_d_ball, FCF_ratio_d_pitch, FCF_ratio_alpha, save_path, output_image)


    return dataset_f_features_all,f_name,labels


#DWT_Energy_Entropy
def wrcoef(signal, coef_type,name, level):
    N = len(signal)
    w = pywt.Wavelet(name)
    a = signal
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

    if N < 1:
        abort(400, "The level must be a positive integer.")

    data = signal
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
        P = -(p * np.log(p+0.0000001))
        energyE = energyE + (p * np.log(p+0.0000001))

    return -energyE

# DWT_Singular_Entropy
def DWT_Singular_Entropy_main(signal,name,N):
    '''

    :param signal: time domain signal
    :param name: Select wavelet function
    :param N: The number of layers of wavelet transform
    :return: WSE; Singular entropy of the signal
    '''

    if N<1:
        abort(400,"The level must be a positive integer.")

    data = signal
    # Calculate the singular values of the signal for each frequency bin
    lamda1 = []
    for i in range(1, N + 1):
        D = wrcoef(data, 'd', name, level=i)
        svd=np.linalg.svd(D, compute_uv=False)
        lamda1.append(svd)

    A = wrcoef(data,'a',name, level=N)
    lamda2 = np.linalg.svd(A,compute_uv=False)

    lamda1.append(lamda2)
    lamda=lamda1

    lamda_total=0
    for i in range(len(lamda)):
        lamda_total=np.sum(lamda[i])
    # Calculate the singular entropy of the signal
    L = len(lamda)
    wse = 0
    for i in range(L):
        p= lamda[i] / lamda_total
        P = -(p * np.log(p+0.0000001))
        wse=np.sum(P)

    return wse

#WaveletPacket_EnergyEntropy
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
        Energy_Entropy[i] = - (P[i]*np.log(P[i]+0.0000001))
    Energy_Entropy_sum = np.sum(Energy_Entropy)

    return Energy_Entropy_sum


# WaveletPacket_Singular_Entropy
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
        Singular_Entropy[i] = -(p[i] * np.log(p[i]+0.0000001))
    Singular_Entropy_sum = np.sum(Singular_Entropy)

    return Singular_Entropy_sum


# OPFCF
def OPFCF_fault_fre(fr, n_ball, d_ball, d_pitch, alpha,fault_type_list):
    """
    Calculate the theoretical fault characteristic frequency
    :return: Current 4 theoretical fault characteristic frequencies
    """
    f_fault = []
    m = d_ball / d_pitch * math.cos(alpha)
    if fault_type_list[0] == 1:  # BPFO
        f_fault0 = (1 - m) * n_ball * fr / 2
        f_fault.append(f_fault0)
    if fault_type_list[1] == 1:  # BPFI
        f_fault1 = (1 + m) * n_ball * fr / 2
        f_fault.append(f_fault1)
    if fault_type_list[2] == 1:  # BSF
        f_fault2 = (1 - m * m) * d_pitch * fr / (2 * d_ball)
        f_fault.append(f_fault2)
    if fault_type_list[3] == 1:  # FTF
        f_fault3 = (1 - m) * fr / 2
        f_fault.append(f_fault3)
    return f_fault


def OPFCF_main(signal, g_var,fault_type_list, fr, order, fs, switch, delta_f0, threshold, k, n_ball, d_ball, d_pitch, alpha):
    '''

    :param signal: input data
    :param fr: rotation frequency
    :param ord: the order of the fault frequency
    :param fs: sampling frequency
    :param fault_type: type of fault, int, 0-FTF, 1-BPFO, 2-BPFI, 3-BSF

    '''

    if threshold<=0:
        abort(400,'The threshold of variance muss be positive.')
    if k <=0 or k >=100:
        abort(400,'The percent muss be in (0, 100).')
    #计算信号的频率范围，以此判断max order范围
    from scipy.fftpack import fft
    fft_y = fft(signal)  # 快速傅里叶变换
    # num = len(signal[0])
    num = len(signal)
    x = np.arange(num)  # 频率个数
    # half_x = x[range(int(num / 2))]  # 取一半区间
    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
    # angle_y = np.angle(fft_y)  # 取复数的角度
    normalization_y = abs_y / num  # 归一化处理（双边频谱）
    normalization_half_y = normalization_y[range(int(num / 2))]  # 由于对称性，只取一半区间（单边频谱）
    findex = np.argwhere(normalization_half_y)
    fmin, fmax = min(findex),max(findex)

    f_fault=OPFCF_fault_fre(fr, n_ball, d_ball, d_pitch, alpha, fault_type_list)
    f_fault_max=float(max(f_fault))
    max_order=int(fmax/f_fault_max)
    if order > max_order or order < 1:
        abort(400, 'The order must be in [1, %d].' % max_order)

    data2 = DataProcessing(signal, fault_type_list, fr, order,  fs, switch, delta_f0, threshold, k, n_ball,
                           d_ball, d_pitch, alpha)
    OP_FCF = data2.probability(g_var)

    return OP_FCF
def OPFCF_RUL_plot(OPFCF_ALL,save_path,output_image=0):
    W=3
    OPFCF_filter_ALL = np.convolve(OPFCF_ALL, np.ones((W,)) / W, mode='valid')
    # 绘制OPFCF—RUL趋势图
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(np.arange(len(OPFCF_ALL)), OPFCF_ALL, color='green', label='Original OPFCF')
    plt.plot(np.arange(len(OPFCF_filter_ALL)), OPFCF_filter_ALL, color='red', label='Filtered OPFCF')
    plt.legend()
    plt.title("OPFCF Trend", fontsize=18)
    plt.xlabel("RUL [%]", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    x_label = np.linspace(0, len(OPFCF_ALL), 101)
    x_label_use = list(range(101))
    x_label_use.reverse()
    plt.xticks(x_label[::5], x_label_use[::5])
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    # plt.ylim(0, 1)
    plt.grid()
    if output_image == 0:
        file_name1 = "%s.png" % "RUL_OPFCF"
        path1 = os.path.join(save_path, file_name1)
        plt.savefig(path1)
    elif output_image == 1:
        file_name1 = "%s.png" % "RUL_OPFCF"
        file_name2 = "%s.jpg" % "RUL_OPFCF"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 2:
        file_name1 = "%s.png" % "RUL_OPFCF"
        file_name2 = "%s.svg" % "RUL_OPFCF"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 3:
        file_name1 = "%s.png" % "RUL_OPFCF"
        file_name2 = "%s.pdf" % "RUL_OPFCF"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    # plt.show()
    plt.close()




# GMR-FFT
def SBN(signal: np.ndarray,data_SBN0):
    """
    Calculate the current actual SBN value.
    :param signal: actual time domain signal
    :return: Actual SBN
    """

    fft_mean = (np.abs(fft(signal)) / len(signal)).mean()
    mean_db = math.log(fft_mean, 10) * 20

    if data_SBN0 is not None:
        if data_SBN0.shape[0] == 1:
            SBN0_fft_mean = (np.abs(fft(data_SBN0)) / len(data_SBN0)).mean()
            SBN0 = math.log(SBN0_fft_mean, 10) * 20
            SBN = SBN0 / mean_db
        else:
            abort(400, 'Input data for SBN0 must be a one dimensional array.')
    else:
        SBN = mean_db

    return SBN


# normal frequency features
def frequency_normal_features(signal,normal_features):
    '''

    :param save_path: path to save
    :param data_source:input data
    :return: 输出为13个频域基本特征，每一组数据样本都包含了一组频域特征（13个）
    '''

    # N=signal.shape[0] # 求得输入数组的行数
    # len=signal.shape[1] # 求得输入数组的列数
    # input_signal = data_source[data_field[3]]
    fft_y = fft(signal)  # 快速傅里叶变换

    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
    # angle_y = np.angle(fft_y)  # 取复数的角度
    # normalization_y = abs_y / N  # 归一化处理（双边频谱）
    # normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）
    frequncy_features=[]
    f_normal_name=[]
    signal_size = np.size(abs_y)
    if normal_features[0]==1:
        frequncy_features_max = np.max(abs_y)
        frequncy_features.append(frequncy_features_max)
        f_normal_name.append('f_max')
    if normal_features[1] == 1:
        frequncy_features_min = np.min(abs_y)
        frequncy_features.append(frequncy_features_min)
        f_normal_name.append('f_min')
    if normal_features[2] == 1:
        frequncy_features_mean = np.mean(abs_y)
        frequncy_features.append(frequncy_features_mean)
        f_normal_name.append('f_mean')
    if normal_features[3] == 1:
        frequncy_features_root_mean_square = np.sqrt(np.mean(abs_y ** 2))
        frequncy_features.append(frequncy_features_root_mean_square)
        f_normal_name.append('f_root_mean_square')
    if normal_features[4] == 1:
        frequncy_features_mean = np.mean(abs_y)
        frequncy_features_standard_deviation = np.sqrt(
        np.sum((abs_y - frequncy_features_mean) ** 2) / (signal_size - 1)) #此处原本是10001
        frequncy_features.append(frequncy_features_standard_deviation)
        f_normal_name.append('f_standard_deviation')
    if normal_features[5] == 1:
        frequncy_features_mean = np.mean(abs_y)
        frequncy_features_variance = np.sum((abs_y - frequncy_features_mean) ** 2) / (
                signal_size - 1)#此处原本是10001
        frequncy_features.append(frequncy_features_variance)
        f_normal_name.append('f_variance')
    if normal_features[6] == 1:
        frequncy_features_median = np.median(abs_y)
        frequncy_features.append(frequncy_features_median)
        f_normal_name.append('f_median')
    if normal_features[7] == 1:
        frequncy_features_mean = np.mean(abs_y)
        frequncy_features_standard_deviation = np.sqrt(
            np.sum((abs_y - frequncy_features_mean) ** 2) / (signal_size - 1))
        frequncy_features_skewness = np.sum(((abs_y - frequncy_features_mean) /
                                         frequncy_features_standard_deviation) ** 3) / signal_size #此处原本是10001
        frequncy_features.append(frequncy_features_skewness)
        f_normal_name.append('f_skewness')
    if normal_features[8] == 1:
        frequncy_features_mean = np.mean(abs_y)
        frequncy_features_standard_deviation = np.sqrt(
            np.sum((abs_y - frequncy_features_mean) ** 2) / (signal_size - 1))
        frequncy_features_kurtosis = np.sum(((abs_y - frequncy_features_mean) /
        frequncy_features_standard_deviation) ** 4) / signal_size
        frequncy_features.append(frequncy_features_kurtosis)
        f_normal_name.append('f_kurtosis')
    if normal_features[9] == 1:
        frequncy_features_max = np.max(abs_y)
        frequncy_features_min = np.min(abs_y)
        frequncy_features_peak_to_peak_value = frequncy_features_max - frequncy_features_min
        frequncy_features.append(frequncy_features_peak_to_peak_value)
        f_normal_name.append('f_peak_to_peak_value')
    if normal_features[10] == 1:
        frequncy_features_root_mean_square = np.sqrt(np.mean(abs_y ** 2))
        frequncy_features_crest_factor = np.max(np.abs(abs_y)) / frequncy_features_root_mean_square
        frequncy_features.append(frequncy_features_crest_factor)
        f_normal_name.append('f_crest_factor')
    if normal_features[11] == 1:
        frequncy_features_root_mean_square = np.sqrt(np.mean(abs_y ** 2))
        frequncy_features_shape_factor = frequncy_features_root_mean_square / np.mean(np.abs(abs_y))
        frequncy_features.append(frequncy_features_shape_factor)
        f_normal_name.append('f_shape_factor')
    if normal_features[12] == 1:
        frequncy_features_impulse_factor = np.max(np.abs(abs_y)) / np.mean(np.abs(abs_y))
        frequncy_features.append(frequncy_features_impulse_factor)
        f_normal_name.append('f_impulse_factor')

    return frequncy_features,f_normal_name


# EMD features
def emd_fault_fre(fr, n_ball, d_ball, d_pitch, alpha,fault_type):
    """
    Calculate the theoretical fault characteristic frequency
    :return: Corresponding theoretical fault characteristic frequency
    """

    m = d_ball / d_pitch * math.cos(alpha)
    if  fault_type == 0:  # BPFO
        f_fault = (1 - m) * n_ball * fr / 2
    elif fault_type == 1:  # BPFI
        f_fault = (1 + m) * n_ball * fr / 2
    elif fault_type == 2:  # BSF
        f_fault = (1 - m * m) * d_pitch * fr /  d_ball / 2
    else:  # FTF
        f_fault = (1 - m) * fr / 2

    return f_fault

def envelope_spectrum(signal, fs):
  """
  1) this function obtains the signal envelope spectrum by Hilbert transform and FFT;
  2) also, the signal has been detrended (both "constant" and "linear") before envelope spectrum extraction
  """
  signal_length = len(signal)
  n_data_num = np.arange(signal_length)
  delta_T = signal_length/fs
  frequency_x = n_data_num/delta_T
  signal_data = detrend(signal, type = 'constant')
  signal_data = detrend(signal, type = 'linear')
  signal_hilbert = np.abs(hilbert(signal_data))
  signal_hilbert = detrend(signal_hilbert, type = 'constant')
  signal_hilbert = detrend(signal_hilbert, type = 'linear')
  envelope_spectrum_y = np.abs(fft(signal_hilbert)/(signal_length/2))
  envelope_spectrum_y[0] = envelope_spectrum_y[0]/2
  envelope_spectrum_y[int(signal_length/2)-1] = envelope_spectrum_y[int(signal_length/2)-1]/2
  frequency_x = frequency_x[0:int(signal_length/2)-1]
  envelope_spectrum_y = envelope_spectrum_y[0:int(signal_length/2)-1]
  return frequency_x, envelope_spectrum_y


def emd_feature(signal_data,fr, n_ball, d_ball, d_pitch, alpha,fs,fault_type,n,ord,limit):
    '''

    :param signal_data: input data
    :param fs: sampling frequency
    :param fault_type: type of fault, int,  0-BPFO, 1-BPFI, 2-BSF,3-FTF
    :param n: range of peak detection (e.g., when n=4, the n-ord fault frequency is the center, the range is 4 sampling points, and the peak detection is performed within this range.)
    :param ord: the order of the fault frequency
    :param limit:Upper limit of frequency range for peak detection (Hz),This is an artificial parameter, because in the envelope spectrum, fault information usually appears in the low frequency part
    :return:
    '''
    if n<1:
        abort(400,"The range of peak detection must be a positive integer.")
    if ord<1:
        abort(400,"The order must be a positive integer.")
    if limit>fs:
        abort(400,"The value of the frequency must be less than or equal to the sampling frequency.")

    bpf=emd_fault_fre(fr, n_ball, d_ball, d_pitch, alpha,fault_type) # failure frequency  (Hz)
    # print(bpf)
    f, p= envelope_spectrum(signal_data, fs)
    value1= np.abs(f[f.any() < limit] - bpf * ord * np.ones([len(f[f.any() < limit]), 1]))
    index = np.argmin(value1)
    if f[index] > bpf * ord:
        if index - int(n / 2) < 0 | index - int(n / 2) == 0:
            # indexb=np.argmax(p[0:index + int(n / 2) - 1])
            b= max(p[0:index + int(n / 2) - 1])
        else:
            # indexb = np.argmax(p[index - int(n / 2):index + int(n / 2) - 1])
            b=max(p[index - int(n / 2):index + int(n / 2) - 1])
            # indexb = indexb - (int(n / 2) + 1) + index

    else:
            if index - int(n / 2) + 1 < 0 | index - int(n / 2) + 1 == 0:
                b=max(p[0:index + int(n / 2) - 1])
                # indexb = np.argmax(p[0:index + int(n / 2) - 1])
            else:
                b=max(p[index - int(n / 2) + 1:index + int(n / 2)])
                # indexb = np.argmax(p[index - int(n / 2) + 1:index + int(n / 2)])
                # indexb = indexb - int(n / 2) + index

    return b

#FCF-ratio:
def FCF_ratio_method(signal, nlevel, order, fs, fr, n_ball, d_ball, d_pitch, alpha):

    if nlevel<1:
        abort(400,'The level must be a positive integer.')
    if order<1:
        abort(400,'The order must be a positive integer.')
    #计算外圈和内圈故障的理论故障频率
    m = d_ball / d_pitch * math.cos(alpha)

    # BPFO 外圈理论故障频率
    BPFO = (1 - m) * n_ball * fr / 2

    # BPFI 内圈理论故障频率
    BPFI = (1 + m) * n_ball * fr / 2

    fc, bw = Fast_Kurtogram(signal, nlevel, fs)
    Wn1 = max([fc - bw / 2, 0.0])
    Wn2 = min([fc + bw / 2, 0.999 * fs / 2])

    #卷积包络谱
    t = np.arange(0, 1, 1/len(signal))
    f0 = (Wn1 + Wn2) / 2
    x = signal - np.mean(signal, 0)
    x0 = x * np.exp(-1j * 2 * np.pi * f0 * t)
    w = firwin(order+1, (Wn2 - Wn1) / 2 / (fs/2))
    xAn = np.convolve(x0, w, mode='same')

    xEnv = np.abs(2 * xAn)
    xEnv = xEnv - np.mean(xEnv, 0)
    Spec = 1 / len(xEnv) * abs(fft(xEnv))

    fSpec = np.arange(0, len(xEnv)) / len(xEnv) * fs
    xSpec = Spec[0:int(len(Spec) / 2) + 1]
    fSpec = fSpec[0:int(len(fSpec) / 2) + 1]
    xSpec[1:len(xSpec)] = 2 * xSpec[1:len(xSpec)]
    envelope_spectrum_y = xSpec
    frequency_x = fSpec
    deltaf = frequency_x[1] - frequency_x[0]
    BPFIAmplitude = max(envelope_spectrum_y[(frequency_x > (BPFI - 5 * deltaf)) & (frequency_x < (BPFI + 5 * deltaf))])
    BPFOAmplitude = max(envelope_spectrum_y[(frequency_x > (BPFO - 5 * deltaf)) & (frequency_x < (BPFO + 5 * deltaf))])
    FCF_ratio=np.log(BPFIAmplitude/BPFOAmplitude)


    return FCF_ratio

def FCF_ratio_plot(dataset,labels, nlevel, order, fs, fr, n_ball, d_ball, d_pitch, alpha,save_path,output_image):
    '''

    :param dataset: mxn,m是样本组数，例如24x12000
    :return:
    '''
    row=dataset.shape[0]

    FCF_ratio_all=[]
    for i in range(row):
        x1 = dataset[i, :].T
        x = np.squeeze(x1)#变成一维
        FCF_ratio = FCF_ratio_method(x, nlevel, order, fs, fr, n_ball, d_ball, d_pitch, alpha)
        FCF_ratio_all.append(FCF_ratio)
    # 绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    num_label = np.unique(labels)

    if len(num_label) > 4:
        abort(400, "There are up to 4 types of bearing failures.")
    if len(labels)<=1:
        abort(400,"For the FCF-ratio diagram, the number of samples muss be greater than 1.")
    if len(labels)>500:
        abort(400,"For a better diagram result, the number of samples should not exceed 500.")
    else:
        labels = np.squeeze(labels)
        for i in range(len(labels)):
            if labels[i] == 0:
                plt.hist(FCF_ratio_all[i], density=True, facecolor="blue", edgecolor="black", alpha=0.7)
            elif labels[i] == 1:
                plt.hist(FCF_ratio_all[i], density=True, facecolor="green", edgecolor="black", alpha=0.7)
            elif labels[i] == 2:
                plt.hist(FCF_ratio_all[i], density=True, facecolor="red", edgecolor="black", alpha=0.7)
            elif labels[i] == 3:
                plt.hist(FCF_ratio_all[i], density=True, facecolor="yellow", edgecolor="black", alpha=0.7)
            else:
                abort(400,"The range of labels for the FCF-ratio diagram is [0,3].")

    # 显示横轴标签
    plt.xlabel('log(BPFIAmplitude / BPFOAmplitude)')
    # plt.xlabel('log(BPFIAmplitude * BPFOAmplitude ^ (-1))')
    # 显示纵轴标签
    plt.ylabel("Numbers")
    # 显示图标题
    plt.title("FCF-ratio distribution histogram")
    if output_image == 0:
        file_name = "%s.png" % "FCF_ratio"
        path = os.path.join(save_path, file_name)
        plt.savefig(path)
    elif output_image == 1:
        file_name1 = "%s.png" % "FCF_ratio"
        file_name2 = "%s.jpg" % "FCF_ratio"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 2:
        file_name1 = "%s.png" % "FCF_ratio"
        file_name2 = "%s.svg" % "FCF_ratio"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 3:
        file_name1 = "%s.png" % "FCF_ratio"
        file_name2 = "%s.pdf" % "FCF_ratio"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)

    # plt.show()
    plt.close()

    return FCF_ratio_all