import scipy.io as sio
import numpy as np
import copy
import os
from moduls.feature_extraction.frequency_feature.load import data_load

from scipy.fftpack import fft,ifft



def frequency_normal_features(input_signal):
    '''

    :param save_path: path to save
    :param data_source:input data
    :return: 输出为13个频域基本特征，每一组数据样本都包含了一组频域特征（13个）
    '''
    # data=data_source['segmentation_data']
    # data_source = sio.loadmat(file_name)
    # data_field = list(data_source.keys())
    # data=data_source[data_field[3]]
    N=input_signal.shape[0] # 求得输入数组的行数
    len=input_signal.shape[1] # 求得输入数组的列数
    # input_signal = data_source[data_field[3]]
    fft_y = fft(input_signal)  # 快速傅里叶变换

    # N = 1400
    # x = np.arange(N)  # 频率个数
    # half_x = x[range(int(N / 2))]  # 取一半区间

    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
    # angle_y = np.angle(fft_y)  # 取复数的角度
    # normalization_y = abs_y / N  # 归一化处理（双边频谱）
    # normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）
    signal_size = np.size(abs_y, 1)
    frequncy_features_max = np.max(abs_y, axis=1)
    frequncy_features_min = np.min(abs_y, axis=1)
    frequncy_features_mean = np.mean(abs_y, axis=1)
    frequncy_features_root_mean_square = np.sqrt(np.mean(abs_y ** 2, axis=1))
    frequncy_features_standard_deviation = np.sqrt(
        np.sum((abs_y - np.tile( frequncy_features_mean, (len, 1)).T) ** 2, axis=1) / (signal_size - 1)) #此处原本是10001
    frequncy_features_variance = np.sum((abs_y - np.tile(frequncy_features_mean, (len, 1)).T) ** 2, axis=1) / (
                signal_size - 1)#此处原本是10001
    frequncy_features_median = np.median(abs_y, axis=1)
    frequncy_features_skewness = np.sum(((abs_y - np.tile(frequncy_features_mean, (len, 1)).T) / np.tile(
        frequncy_features_standard_deviation, (len, 1)).T) ** 3, axis=1) / signal_size #此处原本是10001
    frequncy_features_kurtosis = np.sum(((abs_y - np.tile(frequncy_features_mean, (len, 1)).T) / np.tile(
        frequncy_features_standard_deviation, (len, 1)).T) ** 4, axis=1) / signal_size
    frequncy_features_peak_to_peak_value = frequncy_features_max - frequncy_features_min
    frequncy_features_crest_factor = np.max(np.abs(abs_y), axis=1) / frequncy_features_root_mean_square
    frequncy_features_shape_factor = frequncy_features_root_mean_square / np.mean(np.abs(abs_y), axis=1)
    frequncy_features_impulse_factor = np.max(np.abs(abs_y), axis=1) / np.mean(np.abs(abs_y), axis=1)
    test_output = {"frequncy_features_max": frequncy_features_max, "frequncy_features_min": frequncy_features_min,
                   "frequncy_features_mean": frequncy_features_mean,
                   "frequncy_features_root_mean_square": frequncy_features_root_mean_square,
                   "frequncy_features_standard_deviation": frequncy_features_standard_deviation,
                   "frequncy_features_variance": frequncy_features_variance, "frequncy_features_median": frequncy_features_median,
                   "frequncy_features_skewness": frequncy_features_skewness, "frequncy_features_kurtosis": frequncy_features_kurtosis,
                   "frequncy_features_peak_to_peak_value": frequncy_features_peak_to_peak_value,
                   "frequncy_features_crest_factor": frequncy_features_crest_factor,
                   "frequncy_features_shape_factor": frequncy_features_shape_factor,
                   "frequncy_features_impulse_factor": frequncy_features_impulse_factor}

    frequncy_features = np.concatenate((frequncy_features_max, frequncy_features_min,frequncy_features_mean,frequncy_features_root_mean_square,frequncy_features_standard_deviation,frequncy_features_variance,frequncy_features_median,frequncy_features_skewness,frequncy_features_kurtosis,frequncy_features_peak_to_peak_value,frequncy_features_crest_factor,frequncy_features_shape_factor,frequncy_features_impulse_factor))
    frequncy_features = np.reshape( frequncy_features, (N, 13), order='F')#此处的100根据数据的行数，13根据特征数决定
    # file_name = "frequncy_features.mat"
    # path = os.path.join(save_path, file_name)
    # sio.savemat(path, {'frequncy_features':  frequncy_features},long_field_names=True)
    return  frequncy_features


if __name__ == '__main__':
    # data_source = sio.loadmat(file_name='segmentation_data.mat')
    data = data_load('segmentation_data.mat', "segmentation_data", ".mat", "row")
    frequncy_features=frequency_normal_features(data)
    print(frequncy_features)
