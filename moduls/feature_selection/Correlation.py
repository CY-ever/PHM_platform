import numpy as np
# from load import data_load
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import pandas as pd
from moduls.feature_selection.writein import writein
from moduls.feature_selection.Report_feature_selection import *

from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func


def evaluate_correlation(features,name_list,threshold=0.6,save_path='./', output_file=0, output_image=0):
    '''
    :param features: mxn or nxm, time features or frequency features
    :param labels: mx1 or 1xm,Types of bearing failures
    :param switch: 0:features in raw,1:features in column
    :param save_path: path to save
    :param output_file: type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :param output_image: type to save image,0:png,1:jpg,2:svg,3:pdf
    :return:
    '''

    features = features.T

    # The covariance calculation will report an error when a certain characteristic feature is always constant
    np.seterr(divide='ignore', invalid='ignore')
    correlation_coef = np.abs(np.corrcoef(features))
    correlation_coef = np.vstack(
        (correlation_coef[0:features.shape[0], -1], np.arange(0, features.shape[0], dtype='intp'))).T
    correlation_coef=correlation_coef[:,0] #取出相关系数
    sort=(-correlation_coef).argsort()
    # name_list=name_list[sort]#将降序排列的值对应的特征名排列
    correlation_coef_sorted = correlation_coef[sort]#将值从大到小排列
    #筛选阈值以上的特征

    if threshold>correlation_coef_sorted[0]:
        abort(400,'The maximum threshold is %f.' % correlation_coef_sorted[0])
    else:
        correlation_coef_selected=correlation_coef_sorted[correlation_coef_sorted>=threshold]
        features_new=features[sort][:]
        num=len(correlation_coef_selected)
        # name_list = name_list[sort]  # 将降序排列的值对应的特征名排列
        # name_selected = name_list[:num]
        features_selected=features_new.T[:,:num] #筛选出合适的特征

        # 保存数据
        save_data_func(data=features_selected, output_file=output_file, save_path=save_path,
                       file_name="Correlation",
                       index_label="Downscaled data")

        if name_list is not None:
            name_list = name_list[sort]  # 将降序排列的值对应的特征名排列
            name_selected = name_list[:num]
            #筛选合适的特征
            plt.figure(figsize=(25, 18))
            #筛选特征前出的图
            plt.bar(range(len(name_selected)), correlation_coef_selected, tick_label=name_selected)
            plt.grid(color='b', ls='-.', lw=0.25)
            #筛选特征后出的图
            plt.xticks(rotation=90,fontsize=16)
            plt.yticks(fontsize=16)
            # plt.xlabel('Selected features')
            plt.ylabel("Correlation coefficients",fontsize=18)
            plt.title('Correlation',fontsize=24)
            plt.gcf().subplots_adjust(bottom=0.25)
            if output_image == 0:
                file_name1 = "%s.png" % "feature_selection"
                path1 = os.path.join(save_path, file_name1)
                plt.savefig(path1)
            elif output_image == 1:
                file_name1 = "%s.png" % "feature_selection"
                file_name2 = "%s.jpg" % "feature_selection"
                path1 = os.path.join(save_path, file_name1)
                path2 = os.path.join(save_path, file_name2)
                plt.savefig(path1)
                plt.savefig(path2)
            elif output_image == 2:
                file_name1 = "%s.png" % "feature_selection"
                file_name2 = "%s.svg" % "feature_selection"
                path1 = os.path.join(save_path, file_name1)
                path2 = os.path.join(save_path, file_name2)
                plt.savefig(path1)
                plt.savefig(path2)
            elif output_image == 3:
                file_name1 = "%s.png" % "feature_selection"
                file_name2 = "%s.pdf" % "feature_selection"
                path1 = os.path.join(save_path, file_name1)
                path2 = os.path.join(save_path, file_name2)
                plt.savefig(path1)
                plt.savefig(path2)

            # plt.show()
            plt.close()
        else:
            pass
        return features_selected


__all__ = [
    "evaluate_correlation"
]

if __name__ == "__main__":

    features= writein('features_all.mat',1)
    name=writein('name_all.mat',1)
    # print(name)

    # labels = writein('outer_inner_labels.mat')
    # print(labels)
    a = evaluate_correlation(features,name)
