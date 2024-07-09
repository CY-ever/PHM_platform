import itertools
import numpy as np
from moduls.feature_selection.load import data_load
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import pandas as pd
from moduls.feature_selection.writein import writein
from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func

def evaluate_monotonicity(features,name_list,threshold=0.06,save_path='./',output_file=0,output_image=0):
    '''
    :param features: mxn,二维数组，time features or frequency features
    :param switch: 0:features in raw,1:features in column
    :param save_path: path to save
    :param output_file: type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :param output_image: type to save image,0:png,1:jpg,2:svg,3:pdf
    :return:
    '''
    features = features.T
    monotonicity_list = list()
    for i in range(len(features)):
        a=features[i]
        a_list=[] #取出需要的特征对应的数据 2762*1
        n=len(a)
        monotonicity_old = abs(sum(np.diff(a)>0, 1) - sum(np.diff(a)<0, 1))/(n-1)
        #采用数值方法计算出单调递增以及单调递减的点
        a_list.append(monotonicity_old) #获取每一个特征值的单调性
        features_monotonicity_mean=np.mean(a_list) #求出特征的单调性指标
        monotonicity_list.append(features_monotonicity_mean)
    monotonicity_sorted = np.array(monotonicity_list)
    sort=np.argsort(-monotonicity_sorted) #将变量按照从大到小的顺序进行排序
    monotonicity_new=monotonicity_sorted[sort]#将特征按照数值降序对应排列

    #筛选阈值以上的特征

    if threshold > monotonicity_new[0]:
        abort(400, 'The maximum threshold is %f.' % monotonicity_new[0])
    else:
        monotonicity_selected = monotonicity_new[monotonicity_new >= threshold]
        features_new=features[sort][:]
        num = len(monotonicity_selected)
        features_selected=features_new.T[:,:num] #筛选出合适的特征
        # 保存数据
        save_data_func(data=features_selected, output_file=output_file, save_path=save_path,
                       file_name="Monotonicity",
                       index_label="Downscaled data")

        if name_list is not None:
            name_list = name_list[sort]
            name_selected = name_list[:num]
            #绘图，显示满足阈值的特征
            plt.figure(figsize=(25, 18))
            plt.bar(range(len(name_selected)),  monotonicity_selected, tick_label=name_selected)
            plt.grid(color='b',ls='-.',lw=0.25)
            plt.xticks(rotation=90,fontsize=16)
            plt.yticks(fontsize=16)
            # plt.xlabel('Selected features',fontsize=18)
            plt.ylabel("Monotonicity coefficients",fontsize=18)
            plt.title('Monotonicity',fontsize=24)
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
    "evaluate_monotonicity"
]

if __name__ =="__main__":
    features = writein('features_all.mat', 1)
    name = writein('name_all.mat', 1)
    # print(features.shape,name.shape)
    monotonicity_sorted = evaluate_monotonicity(features,name)
    # print(monotonicity_sorted.shape)
