import numpy as np
from moduls.data_segmentation.load import data_load
import scipy.io as sio
import os
from flask import abort


def data_merge(data1, data2, len):
    data = np.empty([0, 10001])
    merge_data = np.insert(data, 0, values=data1, axis=0)
    merge_data = np.insert(merge_data, 0, values=data2, axis=0)
    # sio.savemat('merge_newdata.mat', {'merge_data': merge_data})
    return merge_data


def signal_segmentaion(save_path, signal, len, N, save_name):
    '''
    :param save_path: path to save
    :param signal:input data
    :param len:新数组的列数
    :param N:如果输入信号的数据行数为1，新数组的行数为N。如果数据为n行，则新数组的行数为n*N

    '''
    if signal.shape[1] <= len and signal.shape[1] != 1:
        abort(400, f"The data segmentaion length must be less than the data length. The current data length is {signal.shape[1]}.")

    print("开始切割")
    sz = signal.shape
    row = sz[0]
    RES = np.empty(shape=(0, len))
    for nn in range(row):
        # print(signal.shape)
        Sig = signal[nn, :]
        L = Sig.shape[0]
        distan = int((L - len) / (N - 1))
        for i in range(N):
            n = i * distan
            en = len + i * distan
            res = signal[nn, n:en]
            res = np.array([res])
            # print(signal[1:5])
            # RES.append(res)
            RES = np.insert(RES, i, values=res, axis=0)
        # print("前RES", RES[1, 0])
        if len == 1:
            RES = RES.astype(np.int)
        # print("后RES", RES[1, 0])
        file_name = f"{save_name}.mat"
        path = os.path.join(save_path, file_name)
        sio.savemat(path, {'data': RES})
    return RES


if __name__ == "__main__":
    # data1 = data_load('data from Chaoren.mat', "acc", ".mat", "row")
    # data2 = data_load('innerring from Chaoren.mat', "acc", ".mat", "row")
    # merge_data=data_merge(data1, data2)
    # RES=signal_segmentaion('./result', merge_data, 200, 50)
    # print(RES.shape)
    # 对1_1_hor西郊数据库进行切分
    # data1 = data_load('1_1_hor.mat', "hor", ".mat", "row")
    # RES = signal_segmentaion('./result', data1, 20000, 200)
    # print(RES.shape)
    # 将西交大数据库第二个工况下的健康数据进行处理合并,得到3x20000
    # data1 = data_load('2_1_hor.mat', "hor", ".mat", "row")
    # data2 = data_load('2_2_hor.mat', "hor", ".mat", "row")
    # # data3 = data_load('2_3_hor.mat', "hor", ".mat", "row")
    # data4 = data_load('2_4_hor.mat', "hor", ".mat", "row")
    # # data5 = data_load('2_5_hor.mat', "hor", ".mat", "row")
    # new_data1=np.empty([0,20000])
    # new_data2 = np.insert(new_data1, 0, values=data1[:,:20000], axis=0)
    # new_data3= np.insert(new_data2, 1, values=data2[:,:20000], axis=0)
    # new_data = np.insert(new_data3, 2, values=data4[:,:20000], axis=0)
    # # new_data5 = np.insert(new_data4, 3, values=data4[:,:20000], axis=0)
    # # new_data = np.insert(new_data5, 4, values=data5[:,:20000], axis=0)
    # file_name = "health_data.mat"
    # path = os.path.join('./result', file_name)
    # sio.savemat(path, {'segmentation_data': new_data})
    # print(new_data.shape)
    # #将内圈故障的文件进行切分，得到3x20000
    # data1 = data_load('2_1_hor.mat', "hor", ".mat", "row")
    # RES = signal_segmentaion('./result', data1[:,-60000:], 20000, 3)
    # print(RES.shape)
    # 将外圈故障文件进行合并
    data2 = data_load('2_2_hor.mat', "hor", ".mat", "row")
    data4 = data_load('2_4_hor.mat', "hor", ".mat", "row")
    data5 = data_load('2_5_hor.mat', "hor", ".mat", "row")
    new_data1 = np.empty([0, 20000])
    new_data2 = np.insert(new_data1, 0, values=data2[:, -20000:], axis=0)
    new_data3 = np.insert(new_data2, 1, values=data4[:, -20000:], axis=0)
    new_data = np.insert(new_data3, 2, values=data5[:, -20000:], axis=0)
    file_name = "outer_data.mat"
    path = os.path.join('./result', file_name)
    sio.savemat(path, {'segmentation_data': new_data})
    print(new_data.shape)

    # 生成一份第二工况下的外圈和内圈的故障数据
    # data1 = data_load('2_1_hor.mat', "hor", ".mat", "row")
    # data2 = data_load('2_2_hor.mat', "hor", ".mat", "row")
    # new_data1=np.empty([0,200000])
    # new_data2 = np.insert(new_data1, 0, values=data1[:,-200000:], axis=0)
    # new_data3= np.insert(new_data2, 1, values=data2[:,-200000:], axis=0)
    # print(new_data3.shape)
    # RES = signal_segmentaion('./result', new_data3, 2000, 100)
    # print(RES.shape)#得到200x2000的数据

    # 生成一份第二工况下的外圈 ，内圈，保持架数据
    # data1 = data_load('2_1_hor.mat', "hor", ".mat", "row")
    # data2 = data_load('2_2_hor.mat', "hor", ".mat", "row")
    # data3 = data_load('2_3_hor.mat', "hor", ".mat", "row")
    # new_data1=np.empty([0,200000])
    # new_data2 = np.insert(new_data1, 0, values=data1[:,-200000:], axis=0)
    # new_data3= np.insert(new_data2, 1, values=data2[:,-200000:], axis=0)
    # new_data = np.insert(new_data3, 2, values=data3[:, -200000:], axis=0)
    # print(new_data.shape)
    # RES = signal_segmentaion('./result', new_data, 2000, 100)
    # print(RES.shape)#得到300x2000的数据
