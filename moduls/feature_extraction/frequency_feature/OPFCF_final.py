# -*- coding: utf-8 -*-

"""

Getting Started -

The class DataProcessing(): This class includes 3 callable function.

-The  DataProcessing.Actual_FCF_Cal(): Real-time time domain signal
frequency domain conversion, and actual FCF feature extraction.

-The  DataProcessing.vargood():  Calculate the variance when the
bearing is intact. Used to assist occurrence probability of fault
characteristic frequency(OPFCF) calculation.

-The  DataProcessing.probability(): Actual OPFCF calculation function


The function def SBN(): Use time domain signal to calculate SBN0 and SBN.


Parameter name settings:
# Define constant
Number of balls: N_BALL
Diameter of balls: D_BALL
Pitch diameter: D_PITCH
The initial contact angle: ALPHA

# Working condition parameters
Sampling Rate(/s): FS
Maximal Order: ORDER
Number of data in each group: NUM

The shaft rotation frequency(in [s]): FR


"""

import math
import numpy as np
import pandas as pd
from scipy.fftpack import fft
from scipy.signal import hilbert, detrend
from flask import abort


class DataProcessing(object):

    def __init__(self, data_use, fault_type_list, fr, order, fs, switch, delta_f0,threshold, k, n_ball, d_ball, d_pitch, alpha):
        """
        :param data_use: actual time domain signal,传进来的是一维数组
        :param fault_type_list: List of bearing failure types:[outer race, inner race, balls, cage]
        """

        self.data_use = data_use.copy()
        self.fault_type_list = fault_type_list
        self.fault_fr_all = self.__fault_fre(fr, n_ball, d_ball, d_pitch, alpha)
        self.fr = fr
        self.order = order
        num=len(data_use)
        self.num = num
        self.fs = fs
        self.switch=switch
        self.delta_f0=delta_f0
        self.threshold = threshold
        self.k = k
        self.n_ball=n_ball
        self.d_ball=d_ball
        self.d_pitch=d_pitch
        self.alpha=alpha


    def var_good(self):
        """
        Calculate the variance when the bearing is intact
        :return: the variance when the bearing is intact
        """
        # data processing
        data_use = self.data_use.copy()
        data_use[0] = 0
        N = len(data_use)
        data_use = np.abs(hilbert(data_use))
        data_use = detrend(data_use)
        data_use[0] = 0
        data_use = np.abs(fft(data_use)) / N
        data_use = data_use[range(int(N / 2))]
        # Calculate the variance of each order when the bearing is intact
        all_var = pd.DataFrame(np.zeros((self.order, len(self.fault_fr_all)), dtype=float))

        min_fault = min(self.fault_fr_all)
        for i in range(self.order):
            for fault in self.fault_fr_all:
                if self.delta_f0 <= 0 or self.delta_f0 >= min_fault:
                    abort(400, "The fixed frequency interval muss be in (0, %f)." % min_fault)
                if self.switch == 0:
                    index_left = int((fault * (i + 1) - self.delta_f0) * self.num / self.fs)
                    index_right = int((fault * (i + 1) + self.delta_f0) * self.num / self.fs)
                elif self.switch == 1:
                    index_left = int((fault * (i + 1) - (self.k/100 * fault * (i + 1))) * self.num / self.fs)
                    index_right = int((fault * (i + 1) + (self.k/100 * fault * (i + 1))) * self.num / self.fs)
                order1 = list(data_use[index_left:index_right + 1])
                g_var = pd.Series(order1).var()
                all_var.iloc[i][self.fault_fr_all.index(fault)] = g_var

        g_var = all_var.mean().mean()
        return g_var

    def probability(self, g_var):
        """
        Actual OPFCF calculation function
        :param g_var:
        :return:
        """
        data_use = self.data_use.copy()

        # data processing
        data_use = np.abs(hilbert(data_use))
        data_use = detrend(data_use)
        data_use[0] = 0
        N = len(data_use)
        data_use = np.abs(fft(data_use)) / N
        data_use = data_use[range(int(N / 2))]

        result = []
        for i in range(self.order):
            for fault in self.fault_fr_all:
                if self.switch == 0:#频域值区间
                    index_left = int((fault * (i + 1) - self.delta_f0) * self.num / self.fs)
                    index_right = int((fault * (i + 1) + self.delta_f0) * self.num / self.fs)
                elif self.switch == 1:#百分比区间
                    index_left = int((fault * (i + 1) - (self.k/100 * fault * (i + 1))) * self.num / self.fs)
                    index_right = int((fault * (i + 1) + (self.k / 100 * fault * (i + 1))) * self.num / self.fs)

                # Calculate the variance within the interval
                var = data_use[index_left:index_right].var()



                if var >= self.threshold * g_var:
                    result.append(1)
                else:
                    result.append(0)

        # Calculate OPFCF
        OP_FCF = sum(result) / (self.order * 4)
        # OP_FCF = sum(result) / (self.order * len(self.fault_fr_all))

        return OP_FCF

    @staticmethod
    def __fault_fre(fr, n_ball, d_ball,d_pitch, alpha):
        """
        Calculate the theoretical fault ault characharacteristic frequency
        :return: Current 4 theoretical fcteristic frequencies
        """
        f_fault = []
        # if fault_type_list is None:
            # f_fault = []
        for type in (0, 1, 2, 3):

            m = d_ball / d_pitch * math.cos(alpha)
            if type == 0:  # BPFO
                f_fault0 = (1 - m) * n_ball * fr / 2
                f_fault.append(f_fault0)
            elif type == 1:  # BPFI
                f_fault1 = (1 + m) * n_ball * fr / 2
                f_fault.append(f_fault1)
            elif type == 2:  # BSF
                f_fault2 = (1 - m * m) * d_pitch * fr / (2 * d_ball)
                f_fault.append(f_fault2)
            else:  # FTF
                f_fault3 = (1 - m) * fr / 2
                f_fault.append(f_fault3)
        return f_fault




