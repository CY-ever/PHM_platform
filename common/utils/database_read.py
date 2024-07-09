import numpy as np
import os
import scipy.io as sio
import pandas as pd
from flask import abort
from utils.save_data import data_save_for_2


def writein(loadpath):
    mat = sio.loadmat(loadpath)
    keys = list(mat.keys())
    print(keys)
    data = mat[keys[3]]
    print("data", data)
    return data


def signal_segmentaion(signal, length, shift):
    '''
    :param signal:input data
    :param len:新数组的列数
    :param N:如果输入信号的数据行数为1，新数组的行数为N。如果数据为n行，则新数组的行数为n*N

    '''
    if shift < 1:
        abort(400, "ERROR: The shift value is too small. shift has a minimum value of 1.")
    raw = signal.shape[0]
    RES = []
    max_length = signal.shape[1]
    print("signal", signal.shape)
    if length <= max_length:
        for i in range(raw):
            signal1 = signal[i]
            for j in range(signal1.size):
                start = j * shift
                end = start + length
                if end <= max_length:
                    res = signal1[start:end]
                    RES.append(res)
                else:
                    break
        try:
            data = np.array(RES)
        except:
            abort(400, "ERROR: The data is too large and out of server memory. Please increase the shift value.")
    else:
        abort(400, "ERROR: The new data is sliced beyond the maximum length %d of the original signal." % max_length)
    return data


def CRWU_read(file_lists=(1, 15), label_lists=(2, 0), file_path="./CRWU", length=2000, shift=500, save_option=False,
              output_file=0, save_path="./result"):
    '''

     :param label_list:
     :param file_lists: Bearing file to select, z.B.(1,) or (1,2,3)
     :param N: Multiplier to increase
     :param len:The length of signal
     :param save_option:to save or not
     :param output_file:Format of the output file

     '''
    if all(i <= 160 and i >= 0 for i in file_lists):
        pass
    else:
        abort(400, "ERROR: The bearing number must be an integer in the range [0, 160]. ")

    if len(label_lists) == 0:
        output_data = []
        list = []
        label0 = (157, 158, 159, 160)
        label1 = (
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 112,
            113,
            114, 115, 116, 117, 118, 119, 120, 121, 122, 123)
        label2 = (
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
            83,
            124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135)
        label3 = (
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
            51,
            84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
            109, 110, 111, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
            154,
            155, 156)
        for i in file_lists:
            if i in label0:
                load_path = os.path.join(file_path, "%d.mat" % i)
                array = writein(load_path)
                array_new = signal_segmentaion(array, length, shift)
                print("切割后形状:", array_new.shape)
                N = array_new.shape[0]
                labels_new = np.zeros(shape=(N, 1))
                list.extend(labels_new)
            elif i in label1:
                load_path = os.path.join(file_path, "%d.mat" % i)
                array = writein(load_path)
                array_new = signal_segmentaion(array, length, shift)
                print("切割后形状:", array_new.shape)
                N = array_new.shape[0]
                labels_new = np.ones(shape=(N, 1))
                list.extend(labels_new)
            elif i in label2:
                load_path = os.path.join(file_path, "%d.mat" % i)
                array = writein(load_path)
                array_new = signal_segmentaion(array, length, shift)
                print("切割后形状:", array_new.shape)
                N = array_new.shape[0]
                labels_new = np.ones(shape=(N, 1)) * 2
                list.extend(labels_new)
            elif i in label3:
                load_path = os.path.join(file_path, "%d.mat" % i)
                array = writein(load_path)
                print("array", array.shape)
                array_new = signal_segmentaion(array, length, shift)
                print("切割后形状:", array_new.shape)
                N = array_new.shape[0]
                labels_new = np.ones(shape=(N, 1)) * 3
                list.extend(labels_new)
            output_data.extend(array_new)
        output_data = np.array(output_data)
        output_labels = np.array(list)
        output_labels = output_labels.astype(int)

    else:
        if len(file_lists) == len(label_lists):
            output_data = []
            output_labels = []
            i = 0
            for file in file_lists:
                load_path = os.path.join(file_path, "%d.mat" % file)
                array = writein(load_path)
                print("array", array.shape)
                array_new = signal_segmentaion(array, length, shift)
                print("切割后形状:", array_new.shape)
                N = array_new.shape[0]
                labels_new = N * [label_lists[i]]
                output_labels.extend(labels_new)
                output_data.extend(array_new)
                i = i + 1
            output_data = np.array(output_data)
            output_labels = np.array(output_labels)
            output_labels = output_labels.reshape(-1, 1)
        else:
            abort(400, "ERROR: The number of labels does not match the number of data.")

    if save_option == True:
        # 选择保存文件的类型
        file_name = "CWRU_data"
        file_name1 = "CWRU_labels"
        data_save_for_2(output_data, output_labels, output_file, save_path, file_name, file_name1,
                        index_label="(sample_nums, sample_length)", index_label1="(sample_nums, labels)")
    else:
        pass
    return output_data, output_labels


def IMS_read(file_lists=(11, 15), label_lists=(), file_path="./IMS", length=2000, shift=5000, function_option=1,
             save_option=True, output_file=0, save_path="./result"):
    '''

    :param file_lists: Bearing file to select, z.B.(1,) or (1,2,3)
    :param length:The length of signal
    :param function_option: Fault diagnosis or RUL prediction. 0:Fault diagnosis, 1:RUL prediction
    :param save_option:to save or not
    :param output_file:Format of the output file

    '''
    if all(i <= 16 and i >= 1 for i in file_lists):
        pass
    else:
        abort(400, "ERROR: The bearing number must be an integer in the range [1, 16].")

    output_data = []
    list = []
    label0 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    label1 = (11, 12)
    label2 = (13, 14)
    label3 = (15, 16)
    if function_option == 0:  # 故障诊断
        if len(label_lists) == 0:
            for i in file_lists:
                load_path = os.path.join(file_path, "%d.mat" % i)
                array = writein(load_path)
                # raw=array.shape[0]
                array_new = signal_segmentaion(array, length, shift)
                N = array_new.shape[0]
                if i in label0:
                    labels_new = np.zeros(shape=(N, 1))
                    list.extend(labels_new)
                elif i in label1:
                    labels_new = np.ones(shape=(N, 1))
                    list.extend(labels_new)
                elif i in label2:
                    labels_new = np.ones(shape=(N, 1)) * 2
                    list.extend(labels_new)
                elif i in label3:
                    labels_new = np.ones(shape=(N, 1)) * 3
                    list.extend(labels_new)
            output_labels = np.array(list)
            output_labels = output_labels.astype(int)
        else:
            if len(file_lists) == len(label_lists):
                output_data = []
                output_labels = []
                i = 0
                for file in file_lists:
                    load_path = os.path.join(file_path, "%d.mat" % file)
                    array = writein(load_path)

                    array_new = signal_segmentaion(array, length, shift)
                    N = array_new.shape[0]
                    labels_new = N * [label_lists[i]]
                    output_labels.extend(labels_new)
                    # output_data.extend(array_new)
                    i = i + 1
                # output_data = np.array(output_data)
                output_labels = np.array(output_labels)
                output_labels = output_labels.reshape(-1, 1)
            else:
                abort(400, "ERROR: The number of labels does not match the number of data.")
    elif function_option == 1:
        for i in file_lists:
            while i in label0:
                abort(400, "ERROR: The selected bearing %d cannot be used for RUL prediction." % i)
                break
            else:
                continue
        for i in file_lists:
            load_path = os.path.join(file_path, "%d.mat" % i)
            array = writein(load_path)
            raw = array.shape[0]
            array_new = signal_segmentaion(array, length, shift)
            N = array_new.shape[0]
            labels_old = np.linspace(1.0, 0, raw)
            labels_new = labels_old.repeat(N / raw)
            list.extend(labels_new)
        output_labels = np.array(list)
        output_labels = output_labels.reshape((-1, 1))

    for file in file_lists:
        load_path = os.path.join(file_path, "%d.mat" % file)
        array = writein(load_path)
        array_new = signal_segmentaion(array, length, shift)
        output_data.extend(array_new)
    output_data = np.array(output_data)
    # print("output",output.shape)

    if save_option == True:
        # 选择保存文件的类型
        file_name = "IMS_data"
        file_name1 = "IMS_labels"
        data_save_for_2(output_data, output_labels, output_file, save_path, file_name, file_name1,
                        index_label="(sample_nums, sample_length)", index_label1="(sample_nums, labels)")

    else:
        pass
    return output_data, output_labels


def paderbron_writein(loadpath, signal_option):
    mat = sio.loadmat(loadpath)
    keys = list(mat.keys())
    if signal_option == 0:
        data = mat[keys[3]]
    elif signal_option == 1:
        data = mat[keys[4]]
    elif signal_option == 2:
        data = mat[keys[5]]
    elif signal_option == 3:
        data = mat[keys[6]]
    elif signal_option == 4:
        data = mat[keys[7]]
    elif signal_option == 5:
        data = mat[keys[8]]
    elif signal_option == 6:
        data = mat[keys[9]]

    return data


def Paderborn_read(file_lists=(27, 28), label_lists=(0, 1), file_path="./Paderborn", length=2000, shift=50,
                   signal_option=6, function_option=0, save_option=False, output_file=0, save_path='./result'):
    '''

     :param file_lists: Bearing file to select, z.B.(1,) or (1,2,3)
     :param N: Multiplier to increase
     :param len:The length of signal
     :param signal_option: which signal to select.0:force, 1:current_1, 2:current_2, 3:speed, 4:temp, 5:torque, 6:vibration
     :param function_option: Fault diagnosis or RUL prediction. 0:Fault diagnosis, 1:RUL prediction
     :param save_option:to save or not
     :param output_file:Format of the output file

     '''
    # if len(label_lists)==0:
    if all(i <= 128 and i >= 1 for i in file_lists):
        pass
    else:
        abort(400, "ERROR: The bearing number must be an integer in the range [1, 128].")

    output_data = []
    list = []
    file_real = (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 65, 66,
        67,
        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110)
    label0 = (27, 28, 29, 30, 31, 32, 59, 60, 61, 62, 63, 64, 91, 92, 93, 94, 95, 96, 123, 124, 125, 126, 127, 128)
    label1 = (
        1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 20, 21, 33, 34, 35, 36, 37, 47, 48, 49, 50, 51, 52, 53, 65, 66, 67, 68, 69,
        79,
        80, 81, 82, 83, 84, 85, 97, 98, 99, 100, 101, 111, 112, 113, 114, 115, 116, 117)
    label2 = (
        9, 10, 11, 12, 13, 14, 22, 23, 24, 25, 26, 41, 42, 43, 44, 45, 46, 54, 55, 56, 57, 58, 73, 74, 75, 76, 77, 78,
        86,
        87, 88, 89, 90, 105, 106, 107, 108, 109, 110, 118, 119, 120, 121, 122)
    if function_option == 0:  # 故障诊断
        if len(label_lists) == 0:
            for i in file_lists:
                load_path = os.path.join(file_path, "%d.mat" % i)
                array = paderbron_writein(load_path, signal_option)
                array_new = signal_segmentaion(array, length, shift)
                N = array_new.shape[0]
                # raw = array.shape[0]
                if i in label0:
                    labels_new = np.zeros(shape=(N, 1))
                    list.extend(labels_new)
                elif i in label1:
                    labels_new = np.ones(shape=(N, 1))
                    list.extend(labels_new)
                elif i in label2:
                    labels_new = np.ones(shape=(N, 1)) * 2
                    list.extend(labels_new)
                else:
                    abort(400, "ERROR: The selected bearing %d cannot be used for fault diagnosis" % i)
            output_labels = np.array(list)
            output_labels = output_labels.astype(int)
        else:
            if len(file_lists) == len(label_lists):
                output_data = []
                output_labels = []
                i = 0
                for file in file_lists:
                    load_path = os.path.join(file_path, "%d.mat" % file)
                    array = paderbron_writein(load_path, signal_option)
                    array_new = signal_segmentaion(array, length, shift)
                    N = array_new.shape[0]
                    labels_new = N * [label_lists[i]]
                    output_labels.extend(labels_new)
                    # output_data.extend(array_new)
                    i = i + 1
                # output_data = np.array(output_data)
                output_labels = np.array(output_labels)
                output_labels = output_labels.reshape(-1, 1)
            else:
                abort(400, "ERROR: The number of labels does not match the number of data.")


    elif function_option == 1:  # RUL预测
        for i in file_lists:
            while i not in file_real:
                abort(400, "ERROR: The selected bearing %d cannot be used for RUL prediction" % i)
                break
            else:
                continue
        for i in file_lists:
            load_path = os.path.join(file_path, "%d.mat" % i)
            array = paderbron_writein(load_path, signal_option)
            array_new = signal_segmentaion(array, length, shift)
            N = array_new.shape[0]
            raw = array.shape[0]
            labels_old = np.linspace(1.0, 0, raw)
            labels_new = labels_old.repeat(N / raw)
            list.extend(labels_new)
        output_labels = np.array(list)
        output_labels = output_labels.reshape((-1, 1))

    for file in file_lists:
        load_path = os.path.join(file_path, "%d.mat" % file)
        array = paderbron_writein(load_path, signal_option)
        array_new = signal_segmentaion(array, length, shift)
        output_data.extend(array_new)
    output_data = np.array(output_data)

    if save_option == True:
        # 选择保存文件的类型
        file_name = "Paderborn_data"
        file_name1 = "Paderborn_labels"
        data_save_for_2(output_data, output_labels, output_file, save_path, file_name, file_name1,
                        index_label="(sample_nums, sample_length)", index_label1="(sample_nums, labels)")

    else:
        pass
    return output_data, output_labels


def FEMTO_read(file_lists=(1, 5), file_path="./FEMTO", length=2000, shift=50, save_option=False, output_file=0,
               save_path="./result"):
    '''

     :param file_lists: Bearing file to select, z.B.(1,) or (1,2,3)
     :param length:The length of signal
     :param save_option:to save or not
     :param output_file:Format of the output file

     '''

    if all(i <= 47 and i >= 1 for i in file_lists):
        pass
    else:
        abort(400, "ERROR: The bearing number must be an integer in the range [1, 47].")

    list = []
    output_data = []
    # file_temp=(3,6,11,14,17,20,23,30,33,36,39,42,47)
    for file in file_lists:
        load_path = os.path.join(file_path, "%d.mat" % file)
        array = writein(load_path)
        raw = array.shape[0]
        array_new = signal_segmentaion(array, length, shift)
        N = array_new.shape[0]
        output_data.extend(array_new)
        labels_old = np.linspace(1.0, 0, raw)
        labels_new = labels_old.repeat(N / raw)
        list.extend(labels_new)
    output_labels = np.array(list)
    output_labels = output_labels.reshape((-1, 1))
    output_data = np.array(output_data)

    if save_option == True:
        # 选择保存文件的类型
        file_name = "FEMTO_data"
        file_name1 = "FEMTO_labels"
        data_save_for_2(output_data, output_labels, output_file, save_path, file_name, file_name1,
                        index_label="(sample_nums, sample_length)", index_label1="(sample_nums, labels)")

    else:
        pass
    return output_data, output_labels


def XJTU_read(file_lists=(1, 7), label_lists=(1, 3), file_path="./XJTU", length=2000, shift=500, function_option=1,
              save_option=True, output_file=0, save_path="./result"):
    '''

     :param file_lists: Bearing file to select, z.B.(1,) or (1,2,3)
     :param len:The length of signal
     :param function_option: Fault diagnosis or RUL prediction. 0:Fault diagnosis, 1:RUL prediction
     :param save_option:to save or not
     :param output_file:Format of the output file

     '''
    if all(i <= 30 and i >= 1 for i in file_lists):
        pass
    else:
        abort(400, "ERROR: The bearing number must be an integer in the range [1, 30].")

    output_data = []
    list = []
    label1 = (1, 2, 3, 4, 5, 6, 13, 14, 17, 18, 19, 20, 21, 22, 29, 30)
    label2 = (11, 12, 25, 26, 27, 28)
    label3 = (7, 8, 15, 16)
    if function_option == 0:  # 故障诊断
        if len(label_lists) == 0:
            for i in file_lists:
                load_path = os.path.join(file_path, "%d.mat" % i)
                array = writein(load_path)
                # raw = array.shape[0]
                array_new = signal_segmentaion(array, length, shift)
                N = array_new.shape[0]
                if i in label1:
                    labels_new = np.ones(shape=(N, 1))
                    list.extend(labels_new)
                elif i in label2:
                    labels_new = np.ones(shape=(N, 1)) * 2
                    list.extend(labels_new)
                elif i in label3:
                    labels_new = np.ones(shape=(N, 1)) * 3
                    list.extend(labels_new)
                else:
                    abort(400, "ERROR: The selected bearing %d cannot be used for fault diagnosis." % i)
            output_labels = np.array(list)
            output_labels = output_labels.astype(int)

        else:
            if len(file_lists) == len(label_lists):
                output_data = []
                output_labels = []
                i = 0
                for file in file_lists:
                    load_path = os.path.join(file_path, "%d.mat" % file)
                    array = writein(load_path)
                    array_new = signal_segmentaion(array, length, shift)
                    N = array_new.shape[0]
                    labels_new = N * [label_lists[i]]
                    output_labels.extend(labels_new)
                    # output_data.extend(array_new)
                    i = i + 1
                # output_data = np.array(output_data)
                output_labels = np.array(output_labels)
                output_labels = output_labels.reshape(-1, 1)
            else:
                abort(400, "ERROR: The number of labels does not match the number of data.")

    elif function_option == 1:  # RUL预测
        for i in file_lists:
            load_path = os.path.join(file_path, "%d.mat" % i)
            array = writein(load_path)
            raw = array.shape[0]
            array_new = signal_segmentaion(array, length, shift)
            N = array_new.shape[0]
            labels_old = np.linspace(1.0, 0, raw)
            labels_new = labels_old.repeat(N / raw)
            list.extend(labels_new)
        output_labels = np.array(list)
        output_labels = output_labels.reshape((-1, 1))

    for file in file_lists:
        load_path = os.path.join(file_path, "%d.mat" % file)
        array = writein(load_path)
        array_new = signal_segmentaion(array, length, shift)
        output_data.extend(array_new)
    output_data = np.array(output_data)

    if save_option == True:
        # 选择保存文件的类型
        file_name = "XJTU_data"
        file_name1 = "XJTU_labels"
        data_save_for_2(output_data, output_labels, output_file, save_path, file_name, file_name1,
                        index_label="(sample_nums, sample_length)", index_label1="(sample_nums, labels)")

    else:
        pass

    print("output_data", output_data)
    return output_data, output_labels


# def data_save_for_2(output_data, output_labels, output_file, save_path, file_name, file_name1):
#     if output_file == 0:
#         file_name = file_name + '.mat'
#         file_name1 = file_name1 + '.mat'
#         path = os.path.join(save_path, file_name)
#         path1 = os.path.join(save_path, file_name1)
#         sio.savemat(path, {'output_data': output_data})
#         sio.savemat(path1, {'output_labels': output_labels})
#     elif output_file == 1:
#         try:
#             dataframe = pd.DataFrame(output_data)
#             file_name = file_name + '.xlsx'
#             file_name1 = file_name1 + '.xlsx'
#             path = os.path.join(save_path, file_name)
#             path1 = os.path.join(save_path, file_name1)
#             writer = pd.ExcelWriter(path)
#             writer1 = pd.ExcelWriter(path1)
#             dataframe.to_excel(writer)
#             dataframe.to_excel(writer1)
#             writer.save()
#             writer1.save()
#         except:
#             abort(400,
#                   f"This sheet is too big! Your worksheet size is: {output_data.shape}. The maximum excel worksheet size is: (1048576, 16384).")
#     elif output_file == 2:
#         file_name = file_name + '.npy'
#         file_name1 = file_name1 + '.npy'
#         path = os.path.join(save_path, file_name)
#         path1 = os.path.join(save_path, file_name1)
#         np.save(path, np.array(output_data))
#         np.save(path1, np.array(output_labels))
#     elif output_file == 3:
#         file_name = file_name + ".csv"
#         file_name1 = file_name1 + ".csv"
#         dataframe = pd.DataFrame(output_data)
#         dataframe1 = pd.DataFrame(output_labels)
#         path = os.path.join(save_path, file_name)
#         path1 = os.path.join(save_path, file_name1)
#         dataframe.to_csv(path, index=False, sep=',')
#         dataframe1.to_csv(path1, index=False, sep=',')
#     elif output_file == 4:
#         file_name = file_name + ".txt"
#         file_name1 = file_name1 + ".txt"
#         path = os.path.join(save_path, file_name)
#         path1 = os.path.join(save_path, file_name1)
#         np.savetxt(path, output_data)
#         np.savetxt(path1, output_labels)
