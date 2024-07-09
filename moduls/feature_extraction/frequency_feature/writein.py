import os
import numpy as np
import xlrd
import pandas as pd
import scipy.io as sio

def writein(loadpath):
    dacl = os.path.splitext(loadpath)[1]
    if '.mat' in dacl:
        mat = sio.loadmat(loadpath)
        keys = list(mat.keys())
        data = mat[keys[3]]

    elif '.npy' in dacl:
        data = np.load(loadpath)
    elif '.xls' in dacl:
        sheet_book = xlrd.open_workbook(loadpath)
        sheet = sheet_book.sheet_by_index(0)
        resArray = []
        for i in range(sheet.nrows):
            line = sheet.row_values(i)
            resArray.append(line)
        data = np.array(resArray)
    elif '.txt' in dacl:
        data = np.loadtxt(loadpath)
    elif '.csv' in dacl:
        data = pd.read_csv(loadpath, header=0)
        data_list = data.values.tolist()
        data_array = np.array(data_list)
        N = int(np.sqrt(len(data_array)))
        data = data_array[-N * N:, 0]
    else:
        print("File format error!")

    return data