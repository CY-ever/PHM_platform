import numpy as np
import pandas as pd
from scipy import io
import os
import xlrd
from flask import abort

import scipy.io as scio
from sklearn.preprocessing import StandardScaler

# 数据切片
def data_slice(data, labels, slice_str="[2,4,6,8;1,2,3,5]"):

    data_shape = data.shape
    # 1. 效验参数有效性
    # 1.是否[]开始和结尾 2.是否以；分割 ,并且
    if (slice_str[0] != '[') or (slice_str[-1] != ']'):
        abort(400, "ERROR: Parameter Data slices can only begin and end with '[]'.")
        # print("数据切片参数必须以'[]'开头或结尾。")
        # return 0, 0
    # 判断中间是否有分号
    if ";" not in slice_str:
        abort(400, "ERROR: Row and column slicing parameters are split by ';'.")
    # 2. 参数处理
    slice_str = slice_str[1:-1]
    # print("slice_str:", slice_str)
    slice_list = slice_str.split(";")
    # print("slice_list:", slice_list)

    # 3. 数据切割
    # 3.1 切割行
    if slice_list[0] != ":":
        # 切割数据
        try:
            slice_list_1 = slice_list[0].split(",")
            # print("slice_list_1", slice_list_1)
            slice_list_1 = [int(i) for i in slice_list_1]
            # print("slice_list_1", slice_list_1)
        except:
            abort(400, "ERROR: Parameter Data slices is not legal. Parameter Data slices can only begin and end with '[]'. Numbers can only be separated by ',', with ':' indicating that rows or columns are not sliced. Row and column slicing parameters are split by ';'. For example: [1,3,5,7,9;:].")
            # print("参数不和规。")
            # return 0, 0

        if max(slice_list_1) >= data.shape[0]:
            abort(400, f"ERROR: Data slice row parameter out of range! The data shape is {data_shape}.")
            # print(f"行超出范围！行最大值为{data.shape[0]-1}。")
            # return 0, 0

        data = data[slice_list_1, :]
        print(data)

        # 切割label
        if labels is not None:
            labels = labels[slice_list_1, :]
            print("labels:", labels)

    # 3.2 切割列
    if slice_list[1] != ":":
        try:
            slice_list_2 = slice_list[1].split(",")
            # print("slice_list_2", slice_list_2)
            slice_list_2 = [int(i) for i in slice_list_2]
            # print("slice_list_2", slice_list_2)
        except:
            abort(400, "ERROR: Parameter Data slices is not legal. Parameter Data slices can only begin and end with '[]'. Numbers can only be separated by ',', with ':' indicating that rows or columns are not sliced. Row and column slicing parameters are split by ';'. For example: [1,3,5,7,9;:].")
            # print("参数不和规。")
            # return 0, 0

        if max(slice_list_2) >= data.shape[1]:
            abort(400, f"ERROR: Data slice column parameter out of range! The data shape is {data_shape}.")
            # print(f"列超出范围！最大为{data.shape[1]-1}。")
            # return 0, 0

        data = data[:, slice_list_2]
        print(data)

    # 返回结果
    return data, labels

def read_time_domain_signal(loadpath, writein_switch=1):
    # dacl = os.path.splitext(loadpath)[1]
    type_list = ["mat", "npy", "xlsx", "txt", "csv"]
    dacl = loadpath.split(".")
    loadpath = ".".join(dacl)
    print("loadpath", loadpath)
    i = 0
    while i<5:
        i += 1
        try:
            if '.mat' in loadpath:
                mat = io.loadmat(loadpath)
                keys = list(mat.keys())
                try:
                    data = mat[keys[4]]  # 此处根据数据源里所提取数据的位置，一般是3或者4
                except:
                    data = mat[keys[3]]  # 此处根据数据源里所提取数据的位置，一般是3或者4
                break
            elif '.npy' in loadpath:
                data = np.load(loadpath)
                break
            elif '.xlsx' in loadpath:
                data = pd.read_excel(loadpath, header=0, index_col=0)
                data_list = data.values.tolist()
                data = np.array(data_list)
                # sheet_book = xlrd.open_workbook(loadpath)
                # sheet = sheet_book.sheet_by_index(0)
                # resArray = []
                # for i in range(sheet.nrows):
                #     line = sheet.row_values(i)
                #     resArray.append(line)
                # data = np.array(resArray)
                # break
            elif '.txt' in loadpath:
                data = np.loadtxt(loadpath)
                break

            elif '.csv' in loadpath:
                print("loadpath.csv", loadpath)
                data = pd.read_csv(loadpath, header=0, index_col=0)
                data_list = data.values.tolist()
                data = np.array(data_list)
                # data = pd.read_csv(loadpath, header=0)
                # data_list = data.values.tolist()
                # data = np.array(data_list)
                # data = data.T
            else:
                data = None
                print("File format error!")
                abort(400, "ERROR: File format error!")
                break
        except:
            try:
                dacl[1] = type_list[i]
            except:
                abort(400, "ERROR: Data read error. Please check that the file has been uploaded.")
            loadpath = ".".join(dacl)
            # abort(400, "Data reading error. Please check the uploaded file.")
            # print("失败,新路径为loadpath", loadpath)
            continue


    print("数据读取,判断前形状:", data.shape)
    if not writein_switch:
        pass
    else:
        data = data.T

    if len(data.shape) == 1:
        data = np.reshape(data, (1, -1))
    print("数据读取,判断后形状:", data.shape)

    print()
    if data.shape[0] == 0 or data.shape[1] == 0:
        # 如数据错误,清楚文件
        del_file_name = loadpath
        os.remove(del_file_name)
        abort(400, f"ERROR: Data read error. The current data shape is {data.shape}. The file has been cleared. Please check the upload data file.")

    return data


def ml_import_data(path, rul_pre, model=None):
    """
    Import data from given path.
    :param path: string, Folder of the data files
    :param model: which model do we use
    :return: ndarray, training and test data
    """
    if not rul_pre:
        # import dataset
        x_train_file = path + 'traindata.mat'
        x_train_dict = scio.loadmat(x_train_file)
        keys = list(x_train_dict.keys())
        x_train_all = x_train_dict.get(keys[3])

        y_train_file = path + 'trainlabel.mat'
        y_train_dict = scio.loadmat(y_train_file)
        keys = list(y_train_dict.keys())
        y_train_all = y_train_dict.get(keys[3])

        x_test_file = path + 'testdata.mat'
        x_test_dict = scio.loadmat(x_test_file)
        keys = list(x_test_dict.keys())
        x_test_all = x_test_dict.get(keys[3])

        y_test_file = path + 'testlabel.mat'
        y_test_dict = scio.loadmat(y_test_file)
        keys = list(y_test_dict.keys())
        y_test_all = y_test_dict.get(keys[3])

        if model in ['CNN', ]:
            size_train = x_train_all.shape[0]
            size_test = x_test_all.shape[0]
            x_train = x_train_all.reshape(size_train, 576, 1)
            x_test = x_test_all.reshape(size_test, 576, 1)
            y_train = to_cat(y_train_all, num_classes=4)
            y_test = to_cat(y_test_all, num_classes=4)
            return x_train, x_test, y_train, y_test

        elif model in ['LSTM', ]:
            size_train = x_train_all.shape[0]
            size_test = x_test_all.shape[0]
            x_train = x_train_all.reshape(size_train, 1, 576)
            x_test = x_test_all.reshape(size_test, 1, 576)
            y_train = to_cat(y_train_all, num_classes=4)
            y_test = to_cat(y_test_all, num_classes=4)
            return x_train, x_test, y_train, y_test

        elif model in ['DBN', 'AE']:
            y_train = to_cat(y_train_all, num_classes=4)
            y_test = to_cat(y_test_all, num_classes=4)
            return x_train_all, x_test_all, y_train, y_test

        elif model in ['SVM', 'KNN', 'RF', "DT", "Bagging", "ET"]:
            # 特征之标准化
            transfer = StandardScaler()
            x_train_all = transfer.fit_transform(x_train_all)
            x_test_all = transfer.fit_transform(x_test_all)
            return x_train_all, x_test_all, y_train_all, y_test_all

        else:
            return x_train_all, x_test_all, y_train_all, y_test_all

    else:
        # import dataset
        x_train_file = path + 'traindata.mat'
        x_train_dict = scio.loadmat(x_train_file)
        keys = list(x_train_dict.keys())
        x_train_all = x_train_dict.get(keys[3])

        y_train_file = path + 'trainlabel.mat'
        y_train_dict = scio.loadmat(y_train_file)
        keys = list(y_train_dict.keys())
        y_train_all = y_train_dict.get(keys[3])

        x_test_file = path + 'testdata.mat'
        x_test_dict = scio.loadmat(x_test_file)
        keys = list(x_test_dict.keys())
        x_test_all = x_test_dict.get(keys[3])

        y_test_file = path + 'testlabel.mat'
        y_test_dict = scio.loadmat(y_test_file)
        keys = list(y_test_dict.keys())
        y_test_all = y_test_dict.get(keys[3])


        if model in ['LSTM', 'CNN']:
            size_train = x_train_all.shape[0]
            size_test = x_test_all.shape[0]
            x_train_all = x_train_all.reshape(size_train, x_train_all.shape[1], 1)
            x_test_all = x_test_all.reshape(size_test, x_test_all.shape[1], 1)
            y_train_all = y_train_all.reshape(size_train, 1)
            y_tes_all = y_test_all.reshape(size_test, 1)

        elif model in ['SVM', 'RF', 'DT', 'DBN', 'AE', ]:
            # 特征之标准化
            size_train = x_train_all.shape[0]
            size_test = x_test_all.shape[0]
            y_train_all = y_train_all.reshape(size_train, 1)
            y_test_all = y_test_all.reshape(size_test, 1)

        elif model in ['KNN', "ET", "Bagging"]:
            pass

        return x_train_all, x_test_all, y_train_all, y_test_all




def to_cat(data, num_classes=None):
    """
    Change the label to One-hot coding.

    :param data: label to change as in [1, 2, 3, 4]
    :param num_classes: total numbers of classes
    :return: Encoded label
    """
    # Each data should be represents the class number of the data
    if num_classes is None:
        num_classes = np.unique(data)
    data_class = np.zeros((data.shape[0], num_classes))
    for i in range(data_class.shape[0]):
        num = data[i]
        data_class[i, num] = 1
    return data_class

def read_data_csv(data_path):
    data = list(np.array(pd.read_csv(data_path)))
    test = []
    for i in range(len(data)):
        if not i % 5:
            try:
                test.append(data[i])
                data.pop(i)
            except:
                continue
    train = np.array(data)
    test = np.array(test)

    y_train = train[:, -1]
    x_train = train[:, :-1]
    y_test = test[:, -1]
    x_test = test[:, :-1]
    print(test)
    y_train = y_train / 2793
    y_test = y_test / 2793

    return x_train, x_test, y_train, y_test