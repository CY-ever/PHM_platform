import numpy as np
import pandas as pd
from moduls.ml.utils import to_cat

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


import os

def del_file(path_data):
    for i in os.listdir(path_data) :    # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "\\" + i    #当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:#os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            continue

if __name__ == '__main__':

    del_file("test")



# if __name__ == '__main__':
#     data_path = "./Dataset/feature_1_1_all.csv"
#     x_train, x_test, y_train, y_test = read_data_csv(data_path)
#     print()
    # data = list(np.array(pd.read_csv("./Dataset/feature_1_1_all.csv")))
    # test = []
    # for i in range(len(data)):
    #     if not i%5:
    #         try:
    #             test.append(data[i])
    #             data.pop(i)
    #         except:
    #             continue
    # train = np.array(data)
    # test = np.array(test)
    #
    # y_train = train[:, -1]
    # x_train = train[:, :-1]
    # y_test = test[:, -1]
    # x_test = test[:, :-1]
    # print(test)
    # return x_train, x_test, y_train, y_test
