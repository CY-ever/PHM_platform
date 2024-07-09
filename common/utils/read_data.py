import os
import numpy as np
from flask import abort

def read_two_data(upload_file):
    from moduls.utils.read_data import read_time_domain_signal
    file_name = 'time_domain_signal.mat'
    label_file_name = 'data_label.mat'
    data_path = os.path.join(upload_file, "upload")
    data_path = os.path.join(data_path, file_name)
    label_path = os.path.join(upload_file, "upload")
    label_path = os.path.join(label_path, label_file_name)
    use_data = read_time_domain_signal(data_path)
    use_data_label = read_time_domain_signal(label_path)
    # use_data = use_data[:5]
    # use_data_label = use_data_label[:5]
    # if use_data.shape[0] != use_data_label.shape[0]:
    #     abort(400, "The data and label are not suitable. Please check the data file again.")

    return use_data, use_data_label


def data_breakdown(data, data_label):
    index = [i for i in range(len(data))]
    np.random.seed(10)
    np.random.shuffle(index)
    use_data = data[index]
    use_data_label = data_label[index]
    # print("模块一输出数据形状为:", use_data.shape, use_data_label.shape)
    return use_data, use_data_label