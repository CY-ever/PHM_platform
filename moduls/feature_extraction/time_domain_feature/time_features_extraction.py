import scipy.io as sio
import numpy as np
import copy
import os
from moduls.feature_extraction.frequency_feature.writein import writein
import pandas as pd


def time_feature_extraction(input_signal,switch=0, save_path='./result',output_file=0):
    '''
    :param data_source:input data，输入为二维数组，例如（200x10000)
    :param switch:当输入数组为mxn时，0：m样本数，1：n样本数
    :param save_path: path to save
    :param output_file: type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :return: 输出为13个时域特征，每一组数据样本都包含了一组时域特征（13个）
    '''
    if switch==0:
        input_signal=input_signal
    elif switch == 1:
        input_signal = input_signal.T
    N=input_signal.shape[0] # 求得输入数组的行数
    len=input_signal.shape[1] # 求得输入数组的列数

    signal_size = np.size(input_signal, 1)
    time_features_max = np.max(input_signal, axis=1)
    time_features_min = np.min(input_signal, axis=1)
    time_features_mean = np.mean(input_signal, axis=1)
    time_features_root_mean_square = np.sqrt(np.mean(input_signal ** 2, axis=1))
    time_features_standard_deviation = np.sqrt(
        np.sum((input_signal - np.tile(time_features_mean, (len, 1)).T) ** 2, axis=1) / (signal_size - 1)) #此处原本是10001
    time_features_variance = np.sum((input_signal - np.tile(time_features_mean, (len, 1)).T) ** 2, axis=1) / (
                signal_size - 1)#此处原本是10001
    time_features_median = np.median(input_signal, axis=1)
    time_features_skewness = np.sum(((input_signal - np.tile(time_features_mean, (len, 1)).T) / np.tile(
        time_features_standard_deviation, (len, 1)).T) ** 3, axis=1) / signal_size #此处原本是10001
    time_features_kurtosis = np.sum(((input_signal - np.tile(time_features_mean, (len, 1)).T) / np.tile(
        time_features_standard_deviation, (len, 1)).T) ** 4, axis=1) / signal_size
    time_features_peak_to_peak_value = time_features_max - time_features_min
    time_features_crest_factor = np.max(np.abs(input_signal), axis=1) / time_features_root_mean_square
    time_features_shape_factor = time_features_root_mean_square / np.mean(np.abs(input_signal), axis=1)
    time_features_impulse_factor = np.max(np.abs(input_signal), axis=1) / np.mean(np.abs(input_signal), axis=1)
    time_features = np.concatenate((time_features_max, time_features_min,time_features_mean,time_features_root_mean_square,time_features_standard_deviation,time_features_variance,time_features_median,time_features_skewness,time_features_kurtosis,time_features_peak_to_peak_value,time_features_crest_factor,time_features_shape_factor,time_features_impulse_factor))
    time_features = np.reshape( time_features, (N, 13), order='F')#此处的N根据数据的行数，13根据特征种类数决定
    # 选择保存文件的类型
    if output_file == 0:
        file_name = 'time_features.mat'
        path = os.path.join(save_path, file_name)
        sio.savemat(path, {'time_features': time_features})
    elif output_file == 1:
        dataframe = pd.DataFrame(time_features)
        file_name = 'time_features.xlsx'
        path = os.path.join(save_path, file_name)
        writer = pd.ExcelWriter(path)
        dataframe.to_excel(writer)
        writer.save()
    elif output_file == 2:
        file_name = 'time_features.npy'
        path = os.path.join(save_path, file_name)
        np.save(path, np.array(time_features))
    elif output_file == 3:
        file_name = "time_features.csv"
        dataframe = pd.DataFrame({'time_features': time_features})
        path = os.path.join(save_path, file_name)
        dataframe.to_csv(path, index=False, sep=',')
    elif output_file == 4:
        file_name = "time_features.txt"
        path = os.path.join(save_path, file_name)
        np.savetxt(path, time_features)

    return  time_features


if __name__ == '__main__':
    #生成外圈和内圈故障数据的时域特征，输入为200x2000
    data = writein('outer_inner_data.mat')
    time_features=time_feature_extraction(data)
    print(time_features.shape)
    #生成外圈内圈保持架的故障数据的时域特征，输入为300x2000
    data = writein('outer_inner_cage_data.mat')
    time_features=time_feature_extraction(data)
    print(time_features.shape)

