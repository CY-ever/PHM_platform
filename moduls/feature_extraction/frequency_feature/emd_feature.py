# Date: 2021.3.23
# Version: 0.2.0
# @author: Yuanheng Mu & Runkai He
# Description: Functions to filter signal with EMD (empirical mode decomposition)

from scipy.signal import argrelextrema
import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt
from moduls.feature_extraction.frequency_feature.load import data_load
from scipy.fftpack import fft
from scipy.signal import hilbert, detrend
import math
from utils.table_setting import *
from utils.save_data import save_data_func
from flask import abort

def envelope_spectrum(signal_data, fs):
  """
  1) this function obtains the signal envelope spectrum by Hilbert transform and FFT;
  2) also, the signal has been detrended (both "constant" and "linear") before envelope spectrum extraction
  """
  signal_length = len(signal_data)
  n_data_num = np.arange(signal_length)
  delta_T = signal_length/fs
  frequency_x = n_data_num/delta_T
  signal_data = detrend(signal_data, type = 'constant')
  signal_data = detrend(signal_data, type = 'linear')
  signal_hilbert = np.abs(hilbert(signal_data))
  signal_hilbert = detrend(signal_hilbert, type = 'constant')
  signal_hilbert = detrend(signal_hilbert, type = 'linear')
  envelope_spectrum_y = np.abs(fft(signal_hilbert)/(signal_length/2))
  envelope_spectrum_y[0] = envelope_spectrum_y[0]/2
  envelope_spectrum_y[int(signal_length/2)-1] = envelope_spectrum_y[int(signal_length/2)-1]/2
  frequency_x = frequency_x[0:int(signal_length/2)-1]
  envelope_spectrum_y = envelope_spectrum_y[0:int(signal_length/2)-1]
  return frequency_x, envelope_spectrum_y

def fault_fre(fr, n_ball, d_ball, d_pitch, alpha,fault_type):
    """
    Calculate the theoretical fault characteristic frequency
    :return: Corresponding theoretical fault characteristic frequency
    """

    m = d_ball / d_pitch * math.cos(alpha)
    if fault_type == 0:  # BPFO
        f_fault = (1 - m) * n_ball * fr / 2
    elif fault_type == 1:  # BPFI
        f_fault = (1 + m) * n_ball * fr / 2
    elif fault_type == 2:  # BSF
        f_fault = (1 - m * m) * d_pitch * fr /  d_ball / 2
    elif fault_type == 3:  # FTF
        f_fault = (1 - m) * fr / 2
    else:
        print("The fault type does not exist.")
        return 0
    return f_fault

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

    bpf=fault_fre(fr, n_ball, d_ball, d_pitch, alpha,fault_type) # failure frequency  (Hz)
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
if __name__ =="__main__":

    n_ball = 16  # Number of balls 8
    d_ball = 22.225  # Diameter of balls 7.92
    d_pitch = np.dot(0.5, (102.7938 + 147.7264))  # Pitch diameter 34.55
    alpha = 0  # the initial contact angle
    fr = 25
    # signal1 = writein('2_GAN_newdata_phy_1000.mat',1)
    # signal = signal1[0]
    # b= emd_feature(signal, fr, n_ball, d_ball, d_pitch, alpha,10000, 0, 3, 3, 2000)
    # print(b)


