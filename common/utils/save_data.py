# from moduls.feature_extraction.frequency_feature.All_frequency_features import *
import scipy.io as sio
import numpy as np
import copy
import os
# from moduls.feature_extraction.time_domain_feature.writein import writein
import pandas as pd
from flask import abort


def save_data_func(data, output_file, save_path, file_name, index_label=""):
    if output_file == 0:
        file_name = file_name + '.mat'
        path = os.path.join(save_path, file_name)
        sio.savemat(path, {file_name: data})

    elif output_file == 1:
        try:
            dataframe = pd.DataFrame(data)
            file_name = file_name + '.xlsx'
            path = os.path.join(save_path, file_name)
            writer = pd.ExcelWriter(path)
            dataframe.to_excel(writer, index=True, index_label=index_label)
            writer.save()
        except:
            abort(400,
                  f"ERROR: This sheet is too big! Your worksheet size is: {data.shape}. The maximum excel worksheet size is: (1048576, 16384).")

    elif output_file == 2:
        file_name = file_name + '.npy'
        path = os.path.join(save_path, file_name)
        np.save(path, np.array(data))

    elif output_file == 3:
        try:
            file_name = file_name + ".csv"
            dataframe = pd.DataFrame(data)
            path = os.path.join(save_path, file_name)
            dataframe.to_csv(path, index=True, index_label=index_label, sep=',')
        except:
            abort(400,
                  f"ERROR: This sheet is too big! Your worksheet size is: {data.shape}. The maximum excel worksheet size is: (1048576, 16384).")

    elif output_file == 4:
        file_name = file_name + ".txt"
        path = os.path.join(save_path, file_name)
        np.savetxt(path, data)


def data_save_for_2(output_data, output_labels, output_file, save_path, file_name, file_name1,
                    index_label="", index_label1="labels"):
    if output_file == 0:
        file_name = file_name + '.mat'
        file_name1 = file_name1 + '.mat'
        path = os.path.join(save_path, file_name)
        path1 = os.path.join(save_path, file_name1)
        sio.savemat(path, {'output_data': output_data})
        sio.savemat(path1, {'output_labels': output_labels})
    elif output_file == 1:
        try:
            dataframe = pd.DataFrame(output_data)
            dataframe1 = pd.DataFrame(output_labels)
            file_name = file_name + '.xlsx'
            file_name1 = file_name1 + '.xlsx'
            path = os.path.join(save_path, file_name)
            path1 = os.path.join(save_path, file_name1)
            writer = pd.ExcelWriter(path)
            writer1 = pd.ExcelWriter(path1)
            dataframe.to_excel(writer, index=True, index_label=index_label)
            dataframe1.to_excel(writer1, index=True, index_label=index_label1)
            writer.save()
            writer1.save()
        except:
            abort(400,
                  f"ERROR: This sheet is too big! Your worksheet size is: {output_data.shape}. The maximum excel worksheet size is: (1048576, 16384).")
    elif output_file == 2:
        file_name = file_name + '.npy'
        file_name1 = file_name1 + '.npy'
        path = os.path.join(save_path, file_name)
        path1 = os.path.join(save_path, file_name1)
        np.save(path, np.array(output_data))
        np.save(path1, np.array(output_labels))
    elif output_file == 3:
        file_name = file_name + ".csv"
        file_name1 = file_name1 + ".csv"
        dataframe = pd.DataFrame(output_data)
        dataframe1 = pd.DataFrame(output_labels)
        path = os.path.join(save_path, file_name)
        path1 = os.path.join(save_path, file_name1)
        dataframe.to_csv(path, index=True, index_label=index_label, sep=',')
        dataframe1.to_csv(path1, index=True, index_label=index_label1, sep=',')
    elif output_file == 4:
        file_name = file_name + ".txt"
        file_name1 = file_name1 + ".txt"
        path = os.path.join(save_path, file_name)
        path1 = os.path.join(save_path, file_name1)
        np.savetxt(path, output_data)
        np.savetxt(path1, output_labels)
