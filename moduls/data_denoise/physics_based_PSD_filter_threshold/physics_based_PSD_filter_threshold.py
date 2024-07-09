import numpy as np
from scipy import signal
# from load import data_load
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import os
import math
import scipy.io as sio
import pandas as pd

from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func


def fault_fre(fr, n_ball, d_ball, d_pitch, alpha):

    m = d_ball / d_pitch * math.cos(alpha)
    # BPFO 外圈理论故障频率
    BPFO = (1 - m) * n_ball * fr / 2

    # BPFI 内圈理论故障频率
    BPFI = (1 + m) * n_ball * fr / 2

    # BSF
    BSF = (1 - m * m) * d_pitch * fr / d_ball

    # FTF 保持架故障频率
    FTF = (1 - m) * fr / 2

    BPFs_list=[BPFO, BPFI, BSF, FTF]
    return BPFs_list

def physics_based_PSD_filter_threshold(fr, n_ball, d_ball, d_pitch, alpha,frequency_band_max, factor, sideband_switch):
  """
  1) to calculate a physcis-based index to determine the threshold of PSD
  2) frequency_band_max is the maximum frequency you are interested in, like 2000Hz or 1000Hz;
  3) BPFs_;list is a list of the first order of different fault characteristic frequencies, [BPFO, BPFI, BSF, FTF];
  4) fr, also shaft_frequency, is the rotation frequency of the bearing shaft, the unit is Hz;
  5) factor is a scalable factor for the user to tune the algorithm;
  6) sideband_switch is a parameter to define what sideband should be considered,
    0: no sideband considered, only BPFO, BPFI, BSF, FTF;
    1: no sideband considered, only BPFO, BPFI, BSF, FTF, fr
    2: only the BPFI's sideband(±fs) considered;
    3: both BPFI's(±fs) and BSF's(±FTF) sideband considered;
  """
  count_test = []
  BPFI_sideband_count = []
  BSF_sideband_count = []
  BPFs_list=fault_fre(fr, n_ball, d_ball, d_pitch, alpha)
  BPFO_1st_order = BPFs_list[0]
  BPFI_1st_order = BPFs_list[1]
  BSF_1st_order = BPFs_list[2]
  FTF_1st_order = BPFs_list[3]
  # to calculate the number of FCF peaks(BPFI, BPFO, BSF, FTF) and shaft frequency peaks
  peak_num_BPFI = int(np.floor(frequency_band_max/BPFI_1st_order))
  peak_num_BPFO = int(np.floor(frequency_band_max/BPFO_1st_order))
  peak_num_BSF = int(np.floor(frequency_band_max/BSF_1st_order))
  peak_num_FTF = int(np.floor(frequency_band_max/FTF_1st_order))
  peak_num_fr = int(np.floor(frequency_band_max/fr))
  for i in range(peak_num_BPFI):
    j = 1
    k = 1
    true_flag_right = 1
    true_flag_left = 1
    while true_flag_right:
      if BPFI_1st_order*(i+1) + j*fr < frequency_band_max:
        BPFI_sideband_count.append(1)
      else:
        true_flag_right = 0
      j += 1
    while true_flag_left:
      if BPFI_1st_order*(i+1) - k*fr > 0:
        BPFI_sideband_count.append(1)
      else:
        true_flag_left = 0
      k += 1
  for i in range(peak_num_BSF):
    j = 1
    k = 1
    true_flag_right = 1
    true_flag_left = 1
    while true_flag_right:
      if BSF_1st_order*(i+1) + j*FTF_1st_order < frequency_band_max:
        BSF_sideband_count.append(1)
      else:
        true_flag_right = 0
      j += 1
    while true_flag_left:
      if BSF_1st_order*(i+1) - k*FTF_1st_order > 0:
        BSF_sideband_count.append(1)
        count_test.append(1)
      else:
        true_flag_left = 0
      k += 1
  if sideband_switch == 0:
    peak_count = peak_num_BPFO + peak_num_BPFI + peak_num_BSF + peak_num_FTF
  if sideband_switch == 1:
    peak_count = peak_num_BPFO + peak_num_BPFI + peak_num_BSF + peak_num_FTF + peak_num_fr
  if sideband_switch == 2:
    peak_count = peak_num_BPFO + peak_num_BPFI + peak_num_BSF + peak_num_FTF + peak_num_fr + sum(BPFI_sideband_count)
  if sideband_switch == 3:
    peak_count = peak_num_BPFO + peak_num_BPFI + peak_num_BSF + peak_num_FTF + peak_num_fr + sum(BPFI_sideband_count) + sum(BSF_sideband_count)
  peak_count = int(np.floor(factor * peak_count))
  return peak_count



def signal_denoise(signal_data, sampling_frequency, cut_off_frequency, filter_method, peak_number, filter_order):
  """
  1) to denoise the signal with power spectrum, and the PSD value is determined with physics-based rules;
  2) signal_data: the signal to denoise
  3) sampling_frequency: the signal sampling frequency, Hz;
  4) cut_off_frequency: the cut_off frequency for the low-pass filter to reduce the high-frequency noise, like 1000Hz;
  5) filter_method: 0: without low-pass filter; 1: with low-pass filter;
  6) peak_number: the number of effective and dominant FCF peaks, which can be determined from function "physics_based_PSD_filter_threshold"
  7) filter_order: the defined order of the low-pass filter, integers, like 6, 7, 8;
  """
  signal_length = len(signal_data)
  omega_n = 2*cut_off_frequency/sampling_frequency
  filter_b, filter_a = signal.butter(filter_order, omega_n, 'low')
  signal_filtered_low_pass = signal.filtfilt(filter_b, filter_a, signal_data)
  if filter_method == 0:
    FFT_signal = np.fft.fft(signal_data, signal_length)
    PSD = FFT_signal * np.conj(FFT_signal)/signal_length
  if filter_method == 1:
    FFT_signal = np.fft.fft(signal_filtered_low_pass, signal_length)
    PSD = FFT_signal * np.conj(FFT_signal)/signal_length
  frequency_x = (sampling_frequency/signal_length) * np.arange(signal_length)
  # index_half = np.arange(1, np.floor(signal_length/2), dtype = np.int32)
  index_half = np.arange(np.floor(signal_length/2), dtype = np.int32)
  PSD_sorted = sorted(PSD, reverse=True)
  PSD_threshold = PSD_sorted[peak_number-1]
  threshold = PSD_threshold
  PSD_index = PSD > threshold
  FFT_filtered = PSD_index * FFT_signal
  signal_filtered = np.fft.ifft(FFT_filtered)
  signal_filtered = np.real(signal_filtered)
  return signal_filtered


if __name__ == "__main__":
  Fs = 25600
  fr = 37.5
  n_ball = 8
  d_ball = 7.92
  d_pitch = 34.55
  alpha = 0
  a= physics_based_PSD_filter_threshold(fr, n_ball, d_ball, d_pitch, alpha,2000, 0.5, 2)
  # signal1 = data_load('image_transformation_DA_newdata.mat', "data", ".mat", "row")
  # signal1 = data_load('outer_data.mat', "segmentation_data", ".mat", "row")
  # signal_data = signal1[0]
  # b=signal_denoise(signal_data, Fs, 1000, 1, a, 6,'./result')
  # a1=physics_based_PSD_filter_threshold(fr, n_ball, d_ball, d_pitch, alpha,2000, 0.5, 3)
  # b1=signal_denoise(signal_data, Fs, 1000, 1, a1, 6,'./result')
  # # print(  signal_data.shape)
  # plot(b, b1)
