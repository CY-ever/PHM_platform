# -*- coding: utf-8 -*-

# -- Sheet --

# Version: 0.1.0
# @author: Runkai He
# Description: load data,load str data

import numpy as np  
import xlrd
import scipy.io as scio


def data_load(raw_data_path,data_name:str, data_type:str,data_struc:str):
    '''
    data_type:".mat" / ".excel" / ".txt"
    data_struc:"row"/"col"
    
    '''
    if (data_type==".txt"):
        raw_data=load_txt_data(raw_data_path)
    elif (data_type==".mat"):
        raw_data=load_mat_data(raw_data_path,data_name)
    elif (data_type==".xlsx"):
        raw_data =load_xlsx_data(raw_data_path)
    else:
        raise Exception("Invalid pass data_type.", data_type)

        
    if (data_struc=="row"):
        pass
    elif(data_struc=="col"):
        raw_data=raw_data.T
    return raw_data

def load_txt_data(raw_data_path):
    data = np.loadtxt(raw_data_path)
    raw_data = data.astype(np.float32)
    return raw_data

def load_mat_data(raw_data_path, date_name):
    raw_data_dict = scio.loadmat(raw_data_path)
    raw_data_array = raw_data_dict[date_name]
    raw_data = np.array(raw_data_array)
    return raw_data

def load_xlsx_data(raw_data_path):
    data = xlrd.open_workbook(raw_data_path) # 打开Excel文件读取数据  # 获取第一个工作表
    table=data.sheet_by_index(0)
    nrows = table.nrows             # 获取行数
    ncols = table.ncols 
    excel_list = np.zeros(shape=(nrows,ncols))
    for col in range (0,ncols):
            for row in range(0,nrows):
                cell_value = table.cell(row, col).value  # 获取单元格数据
                excel_list[row][col]=cell_value # 把数据追加到excel_list中
    raw_data=excel_list
    return raw_data

###str load

def str_load(raw_data_path,data_type:str,data_struc:str):
    '''
    data_type:".mat" / ".excel" 
    data_struc:"row"/"col"
    
    '''
    if (data_type==".mat"):
        str_data=load_mat_str_data(raw_data_path,data_struc)
    elif (data_type==".xlsx"):
        str_data =load_str_data(raw_data_path,data_struc)
    else:
        raise Exception("Invalid pass data_type.", data_type)
    return str_data
        
        
def load_str_data(raw_data_path,data_struc:str):
    data = xlrd.open_workbook(raw_data_path) #   Open the Excel file to read the data and get the first worksheet
    table=data.sheet_by_index(0)
    nrows = table.nrows             # Get the number of rows
    ncols = table.ncols 
    excel_list = np.zeros(shape=(nrows,ncols))
    excel_list=excel_list.astype(np.str)
    for col in range (0,ncols):
            for row in range(0,nrows):
                cell_value = table.cell(row, col).value  # Get cell data
                excel_list[row][col]=cell_value # Append data to excel_list
    if (data_struc=='row'):
        pass
    elif (data_struc=='col'):
        excel_list=excel_list.T
    else:
        raise Exception("Invalid pass data_struc.", data_struc)
    str_data_=[]
    for i in range (ncols):
        a=excel_list[0][i]
        str_data_.append(a)
    str_data=np.array(str_data_)
    return str_data

def load_mat_str_data(raw_data_path,data_struc:str):
    raw_data=load_mat_data(raw_data_path)
    if (data_struc=='row'):
        pass
    elif (data_struc=='col'):
        raw_data=raw_data.T
    else:
        raise Exception("Invalid pass data_struc.", data_struc)
    

__all__ = [
    "data_load",
    "str_load"
    ]

if __name__ == "__main__":
    a = data_load('testdata.mat',"test_data",".mat","row" )
    print(a)