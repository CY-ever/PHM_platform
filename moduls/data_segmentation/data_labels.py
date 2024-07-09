import numpy as np
import scipy.io as sio
import os
def data_labels(fault_type,save_path,m): # 0：外圈，1：内圈，2：保持架
    '''
    :param fault_type: list, e.g.[1,1,1,1]
    :param save_path: path to save
    :param m: 行数，取决于特征值的行数(即样本行数）
    :param n: 列数，取决于特征值的种类数(应该是1）
    :return:
    '''

    label1=np.ones((m,1),dtype=int) # 生成一个m*1的全1数组，因为另一个label为1
    label0=np.zeros((m,1),dtype=int) #生成一个m*1的全0数组，因为label为0
    # label2=np.ones((m, 1),dtype=int) * 2 #生成一个m*1的全2数组，为保持架

    # label2=np.full((m, 1), 2, dtype=int) # 生成一个m*1的值全为2的数组
    # labels=np.vstack((label1,label0,label2)) # 生成2*m行，n列的数组
    labels=np.vstack((label0,label1)) # 生成2*m行，n列的数组
    file_name = "outer_inner_100_labels.mat"
    path = os.path.join(save_path, file_name)
    sio.savemat(path, {'labels': labels})

    return labels

if __name__=="__main__":
    # labels = labels(None,'./result',m=50)
    # print(labels.shape)
    labels = data_labels(None,'./result',m=50) #生成内圈＋外圈的label,各100组
    print(labels.shape)