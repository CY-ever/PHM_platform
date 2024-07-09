import pywt
# from load import data_load
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import pandas as pd
from moduls.data_denoise.DWT.writein import writein
from flask import abort

from utils.table_setting import *
from utils.save_data import save_data_func

def word_create(inputdata,outputdata,name,N,path1,save_path):

    # 1. 创建一个Document对象(docx="word文件路径/None:创建新文档")
    document = Document(docx=None)

    # 2. 创建文件大标题
    document.add_heading("Report of DWT simple filter", level=0)

    # 2.1 运行日志信息
    create_datum = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    create_datum = f"create datum: {create_datum}"

    document.add_paragraph(text=create_datum, style=None)

    """
    注意：
    style = None/"List Bullet"(原点列表)/"List Number"(数字列表)
    """

    # 3. 子标题一: 输入数据的基本信息（如：样本的信息，长度，大小，lable; )
    document.add_heading("1. Input and output for DWT simple filter", level=1)

    # 4. 创建表格
    # 4.1 表格数据准备
    # 注意注意！！！！：表格中内容只支持字符串！！
    table_rows = [
        ("Data", "Data shape"),
        ("input_data", str(inputdata.shape)),
        ("output_data", str(outputdata.shape)),
    ]
    # 4.2 表格对象创建
    table = document.add_table(rows=len(table_rows), cols=len(table_rows[0]))

    # 4.3 循环数据，填入表格
    table_full(table_rows, table)

    # 子标题二：方法的信息：（方法类型，参数配置，）
    # 表格2.1
    document.add_heading("2. Method information", level=1)

    table_rows = [
        ("Parameters", "Parameter Value"),
        ("Wavelet basis", str(name)),
        ("Level", str(N)),
    ]

    table = document.add_table(rows=len(table_rows), cols=len(table_rows[0]))

    table_full(table_rows, table)

    # 子标题三：结果信息；（数据，图，表）
    document.add_heading("3. Results", level=1)

    # 3.2. 放置结果图片
    document.add_heading("Filtered signal", level=2)
    # image_path = os.path.join(save_path, "confusion_matrix.png")
    inline_shape=document.add_picture(image_path_or_stream=path1)
    inline_shape.height= int(document.inline_shapes[0].height * 0.90)
    inline_shape.width = int(document.inline_shapes[0].width * 0.90)

    # 4. word文档保存
    file_name = "Report of DWT simple filter.doc"
    save_path = os.path.join(save_path, file_name)
    document.save(save_path)


def DWT_function(signal, name='db1', N=3,save_path='./',output_file=0,output_image=3):
    '''

    :param save_path: the path to save
    :param signal: input data,(10001,)一维数组
    :param name:name of wavelet transfer
    :param N: number of wavelet decompositions
    :param output_file: type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :param output_image: type to save image,0:png,1:jpg,2:svg,3:pdf

    '''
    # print(signal.shape)
    data = signal.T.tolist()
    w = pywt.Wavelet(name)
    # wavelet transform
    coeffs = pywt.wavedec(data, w, level=N)
    if N > 1:
        # Extraction  Coefficients
        A = coeffs[0]
        D = coeffs[1:]
        # Assign high frequency coefficients to 0
        for n in range(N):
            D[n] = D[n]*0

        Dz=[]
        Dz.append(A.T)
        for n in range(N):
            Dd = D[n].T
            Dz.append(Dd)


        # Reconstruct the filtered wavelet domain signal
        s_rec = pywt.waverec(Dz, name)

    elif N == 1:
        # Extraction Coefficients
        # Assign high frequency coefficients to 0
        A = coeffs[0]
        D = coeffs[1:] * 0
        Dz = []
        Dz.append(A.T)
        Dz.append(D.T)
# Reconstruct the filtered wavelet domain signal
        s_rec = pywt.waverec(Dz, name)
    else:
        abort(400, "ERROR: The level must be a positive integer.")
    #保存数据
    save_data_func(data=s_rec, output_file=output_file, save_path=save_path,
                   file_name="DWT_simple_filter",
                   index_label="Filtered signal")

    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.subplots_adjust(wspace=0, hspace=0.5)
    # plt.figure(figsize=(12, 5))
    plt.title('Raw signal')
    # plt.xlim(8000, 8080)
    plt.xlabel('Sampling points')
    plt.ylabel('Amplitude')
    plt.subplot(2, 1, 2)
    # plt.figure(figsize=(12, 5))
    plt.title('Filtered signal')
    # plt.xlim(8000, 8080)
    plt.xlabel('Sampling points')
    plt.ylabel('Amplitude')
    # plt.ylim(-200, 200)
    plt.plot(s_rec)  # 显示去噪结果
    plt.suptitle('DWT Simple Filter', fontsize=16)
    if output_image==0:
        file_name1 = "%s.png" % "DWT"
        path1 = os.path.join(save_path, file_name1)
        plt.savefig(path1)
    elif output_image==1:
        file_name1 = "%s.png" % "DWT"
        file_name2 = "%s.jpg" % "DWT"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image==2:
        file_name1 = "%s.png" % "DWT"
        file_name2 = "%s.svg" % "DWT"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image==3:
        file_name1 = "%s.png" % "DWT"
        file_name2 = "%s.pdf" % "DWT"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    # plt.show()
    plt.close()
    #生成报告
    word_create(signal, s_rec, name, N, path1, save_path)

    return s_rec

if __name__ =="__main__":
    pass
    # name = pywt.Wavelet('db1')
    # data = writein('image_transformation_DA_newdata1.mat')
    # data = writein('2_GAN_newdata_phy_1000.mat',1)
    # # data = writein('image_transformation_DA_newdata1.mat')
    # print(data.shape)
    # signal = data[0]
    # print(signal.shape)
    # s_rec = DWT_function(signal)
    # print(s_rec.shape)
