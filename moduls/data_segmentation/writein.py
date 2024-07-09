import os
import numpy as np
import xlrd
from scipy import io
import pandas as pd

def writein(loadpath):
    dacl = os.path.splitext(loadpath)[1]
    if '.mat' in dacl:
        mat = io.loadmat(loadpath)
        keys = list(mat.keys())
        data = mat[keys[4]] #此处根据数据源里所提取数据的位置，一般是3或者4
    elif '.npy' in dacl:
        data = np.load(loadpath)
    elif '.xls' in dacl:
        sheet_book = xlrd.open_workbook(loadpath)
        sheet = sheet_book.sheet_by_index(0)
        resArray =[]
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
        data = data_array[-N*N:,0]
    else:
        print("File format error!")
    return data
