import scipy.io as sio
from moduls.data_augmentation.GAN.GAN import GAN_DA

import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from flask import abort

from utils.table_setting import *
from utils.save_data import save_data_func



def GAN_all_main(data,label=None,num=2,Z_dim = 100,save_path='./',output_file=0,output_image=0):
    if num < 1:
        abort(400, "ERROR: The number of augmentations must be a positive integer.")
    if num > 100000:
        abort(400, "ERROR: The number of the increased data set is excessive.")
    if Z_dim <=0:
        abort(400,"ERROR: The input noise size muss be positive.")
    if len(data.shape)==1:
        data=data.reshape(1,-1)

    newdata_all = np.empty(shape=(0, data.shape[1]))
    for i in range(len(data)):
        newdata=GAN_main(data[i],num,Z_dim)
        newdata_all = np.concatenate((newdata_all, newdata), axis=0)
    # 选择data
    save_data_func(data=newdata_all, output_file=output_file, save_path=save_path,
                   file_name="GAN_data", index_label="Augmentation signal")

    if label is None:
        newlabel_all = None
    else:
        label = label.reshape((1, -1))
        newlabel_all = np.empty(shape=(0))
        for i in range(data.shape[0]):
            labels = np.repeat(label[:, i], num + 1, axis=0)
            newlabel_all = np.concatenate((newlabel_all, labels), axis=0)
        newlabel_all = newlabel_all.reshape((-1, 1))  # 将label变为二维
        newlabel_all = newlabel_all.astype(int)
        # 保存label
        save_data_func(data=newlabel_all, output_file=output_file, save_path=save_path,
                       file_name="GAN_label", index_label="(sample_labels)")

    #生成一个个的独立图像
    original_num = 1
    augmentation_num = 1
    for i in range(len(newdata_all)):
        plt.figure(figsize=(12, 5))
        plt.plot(newdata_all[i, :])
        if i%(num+1)==0:
            plt.title("Original Signal"+" "+str(original_num), fontsize=16)
            original_num = original_num + 1
        else:
            plt.title("Augmentation Signal"+" "+str(augmentation_num), fontsize=16)
            augmentation_num = augmentation_num + 1
        plt.xlabel('Sampling points',fontsize=12)
        plt.ylabel('Amplitude',fontsize=12)

        if output_image == 0:
            file_name1 = "GAN_image%d.png" % (i + 1)
            path1 = os.path.join(save_path, file_name1)
            plt.savefig(path1)
        elif output_image == 1:
            file_name1 = "GAN_image%d.png" % (i + 1)
            file_name2 = "GAN_image%d.jpg" % (i + 1)
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        elif output_image == 2:
            file_name1 = "GAN_image%d.png" % (i + 1)
            file_name2 = "GAN_image%d.svg" % (i + 1)
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        elif output_image == 3:
            file_name1 = "GAN_image%d.png" % (i + 1)
            file_name2 = "GAN_image%d.pdf" % (i + 1)
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        # plt.show()
        plt.close()


    return newdata_all,newlabel_all

def GAN_main(data,num,Z_dim):
    '''

    :param num:number to augment
    :param data:input data，一维数组,(10001,)
    :param save_path: path to save
    :param output_file: type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :param output_image: type to save image,0:png,1:jpg,2:svg,3:pdf
    :return:
    '''
    newdata1 = GAN_DA(data, num,Z_dim)
    newdata = np.insert(newdata1, 0, values=data, axis=0)



    return newdata
if __name__=="__main__":
    pass
    # # loadpath='data from Chaoren.mat'
    # # loadpath = 'signal_based_bearing_defect_model_outer.xlsx'
    # loadpath='2_GAN_newdata_phy_1000.mat'
    # data = writein(loadpath,1)
    # # data=data.reshape(1,-1)
    # # label = writein('testlabel.mat')
    # label=None
    # newdata,newlabel_all = GAN_all_main(data[0],label)
