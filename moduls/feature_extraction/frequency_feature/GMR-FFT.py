import math
import numpy as np
import pandas as pd
from scipy.fftpack import fft
from moduls.feature_extraction.frequency_feature.load import data_load
import matplotlib as plt
from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func

def SBN(data_use: np.ndarray,data_SBN0):
    """
    Calculate the current actual SBN value.
    :param data_use: actual time domain signal
    :return: Actual SBN
    """

    fft_mean = (np.abs(fft(data_use)) / len(data_use)).mean()

    mean_db = math.log(fft_mean, 10) * 20

    if data_SBN0 is not None:
        if data_SBN0.ndim==1:
            SBN0_fft_mean=(np.abs(fft(data_SBN0)) / len(data_SBN0)).mean()
            SBN0 = math.log(SBN0_fft_mean, 10) * 20
            SBN = SBN0 / mean_db
        else:
            abort(400,'Input data for SBN0 must be a one dimensional array.')
    else:
        SBN = mean_db

    return SBN


if __name__ == '__main__':
    pass
    # data_use=writein('2_GAN_newdata_phy_1000.mat',1)
    # SBN0=None
    # data_use=data_use[1]
    # print("SBN：", SBN(data_use,SBN0))