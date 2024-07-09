import numpy as np
import os
import scipy.io as sio
from flask import abort


def signal_segmentaion(signal, length, shift):
    '''
    :param signal:input data
    :param len:新数组的列数
    :param N:如果输入信号的数据行数为1，新数组的行数为N。如果数据为n行，则新数组的行数为n*N

    '''
    print("shift:", shift)
    if shift < 1:
        abort(400, "The shift value is too small. shift has a minimum value of 1.")

    if length > signal.shape[1]:
        abort(400, f"The cut sample length is too large. The current data length is {signal.shape[1]}.")
    raw = signal.shape[0]
    RES = []
    max_length = signal.shape[1]
    print("signal", signal.shape)
    if length <= max_length:
        for i in range(raw):
            signal1 = signal[i]
            for j in range(signal1.size):
                start = j * shift
                end = start + length
                if end <= max_length:
                    res = signal1[start:end]
                    RES.append(res)
                else:
                    break
        try:
            data = np.array(RES)
        except:
            abort(400, "The data is too large and out of server memory. Please increase the shift value.")
    else:
        abort(400, "The new data is sliced beyond the maximum length %d of the original signal." % max_length)
    return data
