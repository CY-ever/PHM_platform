#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math


def fault_fre(faulttype, fr, n_ball=8, d_ball=7.92, d_pitch=34.55, alpha=0):
    #     to calculate the fault characteristic frequencies
    #     faulttype: 0-FTF, 1-BPFO, 2-BPFI, 3-BSF
    #     n_ball: the number of balls
    #     d_ball: the diameter of balls
    #     d_pitch: the pitch diameter of bearing
    #     alpha: contact angle
    m = d_ball / d_pitch * math.cos(alpha)
    if faulttype == 0:  # FTF
        f_fault = (1 - m) * fr / 2
    if faulttype == 1:  # BPFO
        f_fault = (1 - m) * n_ball * fr / 2
    if faulttype == 2:  # BPFI
        f_fault = (1 + m) * n_ball * fr / 2
    if faulttype == 3:  # BSF
        f_fault = (1 - m * m) * d_pitch * fr / d_ball
    #     print('a:',f_fault)
    return f_fault


def inputlength(fs, fr, order=5, a=1.2, switch=(1, 1, 1, 1), n_ball=8, d_ball=7.92, d_pitch=34.55, alpha=0):
    """
    计算输入长度
    to calculate the input length
    :return:
    """
    fault = []
    fault.append(fault_fre(0, fr, n_ball, d_ball, d_pitch, alpha))
    if switch[0] == 1:
        fault.append(fault_fre(0, fr, n_ball, d_ball, d_pitch, alpha))
    if switch[1] == 1:
        fault.append(fault_fre(1, fr, n_ball, d_ball, d_pitch, alpha))
    if switch[2] == 1:
        fault.append(fault_fre(2, fr, n_ball, d_ball, d_pitch, alpha))
    if switch[3] == 1:
        fault.append(fault_fre(3, fr, n_ball, d_ball, d_pitch, alpha))


    N1 = fs * a / min(fault)
    #     print(fault)
    #     print(N1)

    f_list = []
    for i in range(1, order + 1):
        f_list += [x * i for x in fault]
    f_list.sort()

    f_min = float("inf")
    for i in range(len(f_list) - 1):
        f_min = min(f_min, abs(f_list[i] - f_list[i + 1]))
    try:
        N2 = fs / f_min
    except:
        N2 = N1

    return int(math.ceil(max(N1, N2)))