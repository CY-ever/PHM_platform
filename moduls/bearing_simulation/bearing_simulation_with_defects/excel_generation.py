# excel_generation.m

## this code was designed to save the output in a .xls file
# author: Diwang Ruan
# date: 2020.09.14
# version: V1.0

import numpy as np
import pandas as pd
import os
import scipy.io as sio
# import scipy.io
# import pandapower as pp


# @function
def excel_generation(acc_x=None, save_path=None,
                     output_file=0, *args, **kwargs):

    # save_path = "bearing_based"

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # if os.path.exists(save_path):
    #     data = pd.DataFrame({
    #         "time(s)": time,
    #         "acc_x(m/s^2)": acc_x,
    #         "acc_y(m/s^2)": acc_y
    #     })
    #
    #     # data.to_excel(save_path, index=False)
    #     data.to_csv(os.path.join(save_path, 'acc.csv'))
    #######
    # if outer_ring_switch == 1:
    if output_file == 0:
        file_name = 'physical_based_model_data.mat'
        path = os.path.join(save_path, file_name)
        sio.savemat(path, {'data': acc_x})
    elif output_file == 1:
        dataframe = pd.DataFrame(acc_x)
        file_name = 'physical_based_model_data.xlsx'
        path = os.path.join(save_path, file_name)
        writer = pd.ExcelWriter(path)
        dataframe.to_excel(writer)
        writer.save()
    elif output_file == 2:
        file_name = 'physical_based_model_data.npy'
        path = os.path.join(save_path, file_name)
        np.save(path, np.array(acc_x))
    elif output_file == 3:
        # print("生成.csv文件")
        print(type(acc_x))
        print(acc_x.shape)
        file_name = "physical_based_model_data.csv"
        # dataframe = pd.DataFrame({'data': acc})
        dataframe = pd.DataFrame(acc_x)
        path = os.path.join(save_path, file_name)
        dataframe.to_csv(path, index=False, sep=',')
    elif output_file == 4:
        file_name = "physical_based_model_data.txt"
        path = os.path.join(save_path, file_name)
        np.savetxt(path, acc_x)

    return


if __name__ == '__main__':
    pass
