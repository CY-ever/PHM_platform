# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 03:16:10 2020
Updated on Dec 21 2020
This script contains functions to perform data augmentation.
Function writein returns input data in different file formats(.mat/.txt/.npy/.xls/.xlsx):
    loadpath: loadpath of input data set (string)
Function noise returns new data with noise with input values:
    x: input signals
    snr: signal to noise ratio(dB)
Function rotation returns new signal after rotating of different angles, with the following input:
    x: input signals
    deg: rotation angles(degree)
    k: signal length
Function left and right return signals with shifting to the left or right, with the following input:
    x: input signals
    k: signal length
    delta: shifting step
Function enlarge and shrink return signals after enlarging or shrinking, with the following input:
    temp1/temp2: input signals
    factor: factor of enlarge and shrink
    k: signal length
Function image_transformation_DA perform data augmentation on each small parts of the signal and then perform data concatenation to return a new data set in form of multi*m*n array(m*n:size of input data set), with the following input:
    loadpath: loadpath of input data set (string)
    multi: multiple of increased data set 
    seg: number of data set's segments 
    column: 0 for row vector, 1 for column vector
    deltax: shifting step in x direction
    deltay: shifting step in y direction
    rot: rotation angle
    noise: signal to noise ratio(dB)
    rescale: factor of enlarge and shrink
    switch: turn on (switch=1) or turn off (switch=0) relative algorithm
@author: Zhang Feifan
@update: Jin Wang/ Yifan Zhao
version: V2.4
"""

import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from flask import abort

from utils.table_setting import *
from utils.save_data import save_data_func

import xlrd
import xlwt
from openpyxl import load_workbook
import openpyxl
# import xlsxwriter



def noise(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


def rotation(x, deg, k):
    y = np.linspace(0, k - 1, k)
    d = np.deg2rad(deg)
    v = np.zeros([2, 2])
    v[0, 0] = np.cos(d)
    v[0, 1] = -np.sin(d)
    v[1, 0] = np.sin(d)
    v[1, 1] = np.cos(d)
    t = np.zeros([2, len(y)])
    t[0, :] = y
    t[1, :] = x
    n = v.dot(t)
    return n[1]


def left(x, k, delta):
    newdata2 = np.zeros(k)
    newdata2[0:k + delta] = x[-delta:k]
    return newdata2


def right(x, k, delta):
    newdata3 = np.zeros(k)
    newdata3[delta:k] = x[0:k - delta]
    return newdata3


def enlarge(temp1, factor, k):
    factor=int(factor)
    en = np.zeros([1, k])
    temp1 = temp1.reshape((1, k))
    for j in range(int(k / factor)):
        for m in range(factor):
            en[0, factor * j + m] = temp1[0, j]
    newdata4 = en * factor
    return newdata4


def shrink(temp2, factor, k):
    shr = np.zeros([1, k])
    temp2 = temp2.reshape((1, k))
    for j in range(int(k * factor) - 1):
        shr[0, j] = temp2[0, int(j / factor)]
    newdata5 = shr * factor
    return newdata5


# 主函数
def image_transformation_DA_main(inputdata,label=None,multi=2, deltax=(-50, 0, 50), deltay=(-0.2, 0.2),
                            rot=(-0.01, -0.015, 0, 0.01, 0.015), snr=(20, 15, 10),
                            rescale=(1 / 3, 1 / 2, 2, 3), switch=(1, 1, 1, 1), save_path='./', output_file=0,
                            output_image=0):
    if multi < 1:
        abort(400,"ERROR: The number of augmentations must be a positive integer.")
    if multi > 100000:
        abort(400, "ERROR: The number of the increased data set is excessive.")
    if switch[3]==1 and rescale<=0:
        abort(400,"ERROR: The scale factor muss be positive.")

    newdata_all = np.empty(shape=(0, inputdata.shape[1]))
    for i in range(inputdata.shape[0]):
        data=inputdata[i]
        newdata=image_transformation_DA(data,multi, deltax, deltay,rot, snr,rescale, switch)
        newdata_all = np.concatenate((newdata_all,newdata), axis=0)
    if label is None:
        newlabel_all=None
    else:
        label = label.reshape((1, -1))
        newlabel_all = np.empty(shape=(0))
        for i in range(inputdata.shape[0]):
            labels = np.repeat(label[:, i], multi + 1, axis=0)
            newlabel_all = np.concatenate((newlabel_all, labels), axis=0)
        newlabel_all = newlabel_all.reshape((-1, 1))  # 将label变为二维
        newlabel_all = newlabel_all.astype(int)
        #保存labels
        save_data_func(data=newlabel_all,output_file=output_file,save_path=save_path,file_name="Image_transformation_label",index_label="Labels")

    # 保存剧增后的数据
    save_data_func(data=newdata_all, output_file=output_file, save_path=save_path, file_name="Image_transformation_data",
                   index_label="Augmentation data")

    #生成一个个的独立图像
    original_num = 1
    augmentation_num = 1
    for i in range(len(newdata_all)):
        if i>=20:
            break
        plt.figure(figsize=(12, 5))
        plt.plot(newdata_all[i,:])
        if i%(multi+1)==0:
            plt.title("Original Signal"+" "+str(original_num),fontsize=16)
            original_num = original_num + 1
        else:
            plt.title("Augmentation Signal"+" "+str(augmentation_num), fontsize=16)
            augmentation_num = augmentation_num + 1

        plt.xlabel('Sampling points',fontsize=12)
        plt.ylabel('Amplitude',fontsize=12)
        if output_image == 0:
            file_name1 = "transformation_DA_image_new%d.png" % (i + 1)
            path1 = os.path.join(save_path, file_name1)
            plt.savefig(path1)
        elif output_image == 1:
            file_name1 = "transformation_DA_image_new%d.png" % (i + 1)
            file_name2 = "transformation_DA_image%d.jpg" % (i + 1)
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        elif output_image == 2:
            file_name1 = "transformation_DA_image_new%d.png" % (i + 1)
            file_name2 = "transformation_DA_image%d.svg" % (i + 1)
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        elif output_image == 3:
            file_name1 = "transformation_DA_image_new%d.png" % (i + 1)
            file_name2 = "transformation_DA_image%d.pdf" % (i + 1)
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)

        # plt.show()
        plt.close()


    return newdata_all,newlabel_all

def image_transformation_DA(data,multi, deltax, deltay,rot, snr,rescale, switch):
    seg = 4
    data = data.astype(np.float32)
    data=data.reshape((1,-1))#将输入统一变为二维数组
    k = int(data.shape[1] / seg)
    newdata = np.zeros((multi, data.shape[0], data.shape[1]))

    li = []
    if switch[0] == 1:
        li.append('rotation')
    if switch[1] == 1:
        li.append('noise')
    if switch[2] == 1:
        li.append('trans')
    if switch[3] == 1:
        li.append('rescale')


    for i in range(data.shape[0]):
        for num in range(multi):
            for parts in range(seg):
                r = random.randint(0, len(li) - 1)
                operator = li[r]
                if operator == 'rotation':
                    par = random.randint(0, len(rot) - 1)
                    newdata[num, i, k * parts:k * (1 + parts)] = rotation(data[i, k * parts:k * (1 + parts)], rot[par],
                                                                          k)
                elif operator == 'noise':
                    par = random.randint(0, len(snr) - 1)
                    newdata[num, i, k * parts:k * (1 + parts)] = noise(data[i, k * parts:k * (1 + parts)],
                                                                       snr[par]) + data[i, k * parts:k * (1 + parts)]
                elif operator == 'trans':
                    parx = random.randint(0, len(deltax) - 1)
                    pary = random.randint(0, len(deltay) - 1)
                    if deltax[parx] < 0:
                        newdata[num, i, k * parts:k * (1 + parts)] = left(data[i, k * parts:k * (1 + parts)], k,
                                                                          deltax[parx]) + deltay[pary]
                    elif deltax[parx] > 0:
                        newdata[num, i, k * parts:k * (1 + parts)] = right(data[i, k * parts:k * (1 + parts)], k,
                                                                           deltax[parx]) + deltay[pary]
                    else:
                        newdata[num, i, k * parts:k * (1 + parts)] = data[i, k * parts:k * (1 + parts)] + deltay[pary]
                elif operator == 'rescale':
                    par = random.randint(0, len(rescale) - 1)
                    if rescale[par] < 1:
                        newdata[num, i, k * parts:k * (1 + parts)] = shrink(data[i, k * parts:k * (1 + parts)],
                                                                            rescale[par], k)
                    elif rescale[par] > 1:
                        newdata[num, i, k * parts:k * (1 + parts)] = enlarge(data[i, k * parts:k * (1 + parts)],
                                                                             rescale[par], k)
                    else:
                        newdata[num, i, k * parts:k * (1 + parts)] = data[i, k * parts:k * (1 + parts)]
                else:
                    break

    newdata=np.reshape(newdata,(multi,data.shape[1]))
    newdata=np.insert(newdata, 0, values=data, axis=0)

    return newdata


if __name__ == "__main__":
    pass
    # # data = writein('data from Chaoren.mat')
    # # data=writein('signal_based_bearing_defect_model_outer.csv')
    # data = writein('2_GAN_newdata_phy_1000.mat',1)
    # data1 = data[0]
    # data2 = data1.reshape((1, -1))
    # # label=writein('testlabel.mat')
    # multi = 3
    # label=None
    # newdata_all,labels=image_transformation_DA_main(data2,label,multi,rot=(-0.005,-0.05),save_path='./')
    # print(newdata_all.shape)
    # # newdata_all=image_transformation_DA(data[0])
    # # print(labels)



