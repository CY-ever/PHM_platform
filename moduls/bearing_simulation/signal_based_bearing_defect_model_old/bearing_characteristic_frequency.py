## this code was designed to calculate the characteristic frequencies for bearing, like: BPFO, BPFI,BSF.
# author: Diwang Ruan
# date: 2020.09.04
# version: V1.0

# once bearing has a defect on the outer ring, inner ring or balls, we could identify peaks at corresponding defect frequency and its harmonics from the acceleration spectrum
# BPFO: ball pass frequency of outer race
# BPFI: ball pass frequency of inner race
# BSF:  ball spin frequency


## the input and output of the function:
# the 1st input x of this function has 5 elements as follows
# input elements should be assigned with values as the following order
# D: ball diameter, unit:mm;
# d: pitch diameter, unit: mm;
# alpha: bearing free contact angle, unit: deg;
# Z: number of rolling elements, unit: /;
# fs: shaft rotation frequency, unit: Hz;
# the 2nd input defect_type is used to identify which BPFs should be output
# 1: BPFO; 2: BPFI; 3: BSF
# the output of this function has three elements: [BPFO,BPFI,BSF]

## the code for the main function

import numpy as np


def bearing_characteristic_frequency(x=None, defect_type=None, *args, **kwargs):
    # varargin = bearing_characteristic_frequency.varargin
    # nargin = bearing_characteristic_frequency.nargin

    # assignment for main parameters
    D = x[0]
    # bearing_characteristic_frequency.m:29
    d = x[1]
    # bearing_characteristic_frequency.m:30
    alpha = np.deg2rad(x[2])
    # bearing_characteristic_frequency.m:31
    n = x[3]
    # bearing_characteristic_frequency.m:32
    fs = x[4]
    # bearing_characteristic_frequency.m:33
    # formulations to calculate BPFO, BPFI and BSF.
    BPFO = np.dot(np.dot(n, fs) / 2, (1 - np.dot(D / d, np.cos(alpha))))
    # bearing_characteristic_frequency.m:36
    BPFI = np.dot(np.dot(n, fs) / 2, (1 + np.dot(D / d, np.cos(alpha))))
    # bearing_characteristic_frequency.m:37
    BSF = np.dot(np.dot(d / D, fs) / 2, (1 - (np.dot(D / d, np.cos(alpha))) ** 2))
    # bearing_characteristic_frequency.m:38
    # to determine the final output by "defect_type".
    frequencies = [BPFO, BPFI, BSF]
    # bearing_characteristic_frequency.m:41
    defect_frequency = frequencies[defect_type]
    # bearing_characteristic_frequency.m:42
    print('defect_frequency is:', defect_frequency)
    return defect_frequency


if __name__ == '__main__':
    # print(bearing_characteristic_frequency())

    pass
