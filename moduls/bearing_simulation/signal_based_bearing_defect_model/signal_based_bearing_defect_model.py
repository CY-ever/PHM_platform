## this code was developed to model the acceleration amplitude under different defect type
# author: Diwang Ruan
# date: 2020.09.18
# version: V1.0

## brief introduction of the signal-based bearing defect simulation code package
# main function: 
# to model the accleration respoonse of bearing with single defect on outer ring, inner ring or ball
# structure: 
# "signal_based_bearing_defect_model" is the main function;
# "amplitude" is called by "signal_based_bearing_defect_model" to calculate the acceleration amplitude;
# "exponential_decay" is called by "signal_based_bearing_defect_model" to simulate the decay process of of each strike impulse��
# in the definition of defect_parameter, "bearing_characteristic_frequcy" is called to calculate the BPFO, BPFI and BSF with bearing geometric and condition parameters.
# input and output:
# input: the inputs are 4 structures defined as below: bearing_parameter, condition_parameter, defect_parameter, sim_parameter;
# output��[time, acc]: time array(s) and acceleration array(m/s^2) during the whole simulation duration;

# the main code for "signal_based_bearing_defect_model"

import matplotlib.pyplot as plt
import numpy as np
from moduls.bearing_simulation.signal_based_bearing_defect_model.amplitude import amplitude as am
from moduls.bearing_simulation.signal_based_bearing_defect_model.exponential_decay import exponential_decay
import os
import scipy.io as sio
import pandas as pd
# from module1.signal_segmentaion import signal_segmentaion


def signal_based_bearing_defect_model(bearing_parameter=None, defect_parameter=None, condition_parameter=None,
                                      sim_parameter=None, save_path=None, output_file=0, output_image=0):
    # varargin = signal_based_bearing_defect_model.varargin
    # nargin = signal_based_bearing_defect_model.nargin

    # assignment for main parameters
    step_size = sim_parameter.step_size

    duration = sim_parameter.duration

    defect_frequency = defect_parameter.defect_frequency

    # definition of the output size
    output_size = len(np.arange(0, duration, step_size))

    # x_out = np.zeros(1, output_size)
    x_out = np.zeros(output_size)

    # to calculate how many impulses will be generated during the duration
    num_impulse = np.floor(duration / (1 / defect_frequency))

    # the main iteration loop with defined step_size
    for sim_time in np.arange(0, duration, step_size).reshape(-1):
        # Am = am.amplitude(sim_time, bearing_parameter, condition_parameter, defect_parameter)
        Am = am(sim_time, bearing_parameter, condition_parameter, defect_parameter)

        x_temp = 0

        for k in np.arange(1, num_impulse).reshape(-1):
            # x = exponential_decay.exponential_decay(sim_time, k, defect_parameter, condition_parameter)
            x = exponential_decay(sim_time, k, defect_parameter, condition_parameter)
            x_temp = x_temp + np.dot(Am, x)

        x_out[int(sim_time / step_size)] = x_temp

    time = np.arange(0, duration, step_size)

    acc = np.copy(x_out)

    # plot for time-acc

    # title_string = ['outer ring', 'inner ring', 'ball']
    #
    # title_temp = 'acceleration of bearing with defect on ' + title_string[defect_parameter.defect_type]
    # plt.plot(np.arange(0, duration, step_size), x_out)
    # plt.title(title_temp)
    #
    # plt.xlabel('time[s]')
    #
    # plt.ylabel('acceleration[m·s^-2]')

    # if save_path:
    # save_path = "signal_based_figure"

    # name_list = ["outer", "inner", "ball"]

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # if os.path.exists(save_path):
        # for i in range(all_data.shape[0]):
        #     plt.plot(np.arange(0, duration, step_size), all_data[i, :])
        #     plt.title(title_temp)
        #
        #     plt.xlabel('time[s]')
        #
        #     plt.ylabel('acceleration[m·s^-2]')
        #     if output_image == 0:
        #         file_name = "%s.png" % "signal_based_bearing_defect_model"
        #         path = os.path.join(save_path, file_name)
        #         plt.savefig(path)
        #     elif output_image == 1:
        #         file_name1 = "%s.png" % "signal_based_bearing_defect_model"
        #         file_name2 = "%s.jpg" % "signal_based_bearing_defect_model"
        #         path1 = os.path.join(save_path, file_name1)
        #         path2 = os.path.join(save_path, file_name2)
        #         plt.savefig(path1)
        #         plt.savefig(path2)
        #     elif output_image == 2:
        #         file_name1 = "%s.png" % "signal_based_bearing_defect_model"
        #         file_name2 = "%s.svg" % "signal_based_bearing_defect_model"
        #         path1 = os.path.join(save_path, file_name1)
        #         path2 = os.path.join(save_path, file_name2)
        #         plt.savefig(path1)
        #         plt.savefig(path2)
        #     elif output_image == 3:
        #         file_name1 = "%s.png" % "signal_based_bearing_defect_model"
        #         file_name2 = "%s.pdf" % "signal_based_bearing_defect_model"
        #         path1 = os.path.join(save_path, file_name1)
        #         path2 = os.path.join(save_path, file_name2)
        #         plt.savefig(path1)
        #         plt.savefig(path2)
        #     plt.close()

        # if defect_parameter.defect_type == 0:
        #     if output_image == 0:
        #         file_name = "%s.png" % "signal_based_bearing_defect_model"
        #         path = os.path.join(save_path, file_name)
        #         plt.savefig(path)
        #     elif output_image == 1:
        #         file_name1 = "%s.png" % "signal_based_bearing_defect_model"
        #         file_name2 = "%s.jpg" % "signal_based_bearing_defect_model"
        #         path1 = os.path.join(save_path, file_name1)
        #         path2 = os.path.join(save_path, file_name2)
        #         plt.savefig(path1)
        #         plt.savefig(path2)
        #     elif output_image == 2:
        #         file_name1 = "%s.png" % "signal_based_bearing_defect_model"
        #         file_name2 = "%s.svg" % "signal_based_bearing_defect_model"
        #         path1 = os.path.join(save_path, file_name1)
        #         path2 = os.path.join(save_path, file_name2)
        #         plt.savefig(path1)
        #         plt.savefig(path2)
        #     elif output_image == 3:
        #         file_name1 = "%s.png" % "signal_based_bearing_defect_model"
        #         file_name2 = "%s.pdf" % "signal_based_bearing_defect_model"
        #         path1 = os.path.join(save_path, file_name1)
        #         path2 = os.path.join(save_path, file_name2)
        #         plt.savefig(path1)
        #         plt.savefig(path2)
        #     plt.close()

            # if output_file == 0:
            #     file_name = 'signal_based_bearing_defect_model_outer.mat'
            #     path = os.path.join(save_path, file_name)
            #     sio.savemat(path, {'data': acc})
            # elif output_file == 1:
            #     dataframe = pd.DataFrame(acc)
            #     file_name = 'signal_based_bearing_defect_model_outer.xlsx'
            #     path = os.path.join(save_path, file_name)
            #     writer = pd.ExcelWriter(path)
            #     dataframe.to_excel(writer)
            #     writer.save()
            # elif output_file == 2:
            #     file_name = 'signal_based_bearing_defect_model_outer.npy'
            #     path = os.path.join(save_path, file_name)
            #     np.save(path, np.array(acc))
            # elif output_file == 3:
            #     print("生成.csv文件")
            #     print(type(acc))
            #     print(acc.shape)
            #     file_name = "signal_based_bearing_defect_model_outer.csv"
            #     # dataframe = pd.DataFrame({'data': acc})
            #     dataframe = pd.DataFrame(acc)
            #     path = os.path.join(save_path, file_name)
            #     dataframe.to_csv(path, index=False, sep=',')
            # elif output_file == 4:
            #     file_name = "signal_based_bearing_defect_model_outer.txt"
            #     path = os.path.join(save_path, file_name)
            #     np.savetxt(path, acc)

        # elif defect_parameter.defect_type == 1:
        #     if output_image == 0:
        #         file_name = "%s.png" % "signal_based_bearing_defect_model"
        #         path = os.path.join(save_path, file_name)
        #         plt.savefig(path)
        #     elif output_image == 1:
        #         file_name1 = "%s.png" % "signal_based_bearing_defect_model"
        #         file_name2 = "%s.jpg" % "signal_based_bearing_defect_model"
        #         path1 = os.path.join(save_path, file_name1)
        #         path2 = os.path.join(save_path, file_name2)
        #         plt.savefig(path1)
        #         plt.savefig(path2)
        #     elif output_image == 2:
        #         file_name1 = "%s.png" % "signal_based_bearing_defect_model"
        #         file_name2 = "%s.svg" % "signal_based_bearing_defect_model"
        #         path1 = os.path.join(save_path, file_name1)
        #         path2 = os.path.join(save_path, file_name2)
        #         plt.savefig(path1)
        #         plt.savefig(path2)
        #     elif output_image == 3:
        #         file_name1 = "%s.png" % "signal_based_bearing_defect_model"
        #         file_name2 = "%s.pdf" % "signal_based_bearing_defect_model"
        #         path1 = os.path.join(save_path, file_name1)
        #         path2 = os.path.join(save_path, file_name2)
        #         plt.savefig(path1)
        #         plt.savefig(path2)
        #     plt.close()

            # if output_file == 0:
            #     file_name = 'signal_based_bearing_defect_model_inner.mat'
            #     path = os.path.join(save_path, file_name)
            #     sio.savemat(path, {'data': acc})
            # elif output_file == 1:
            #     dataframe = pd.DataFrame(acc)
            #     file_name = 'signal_based_bearing_defect_model_inner.xlsx'
            #     path = os.path.join(save_path, file_name)
            #     writer = pd.ExcelWriter(path)
            #     dataframe.to_excel(writer)
            #     writer.save()
            # elif output_file == 2:
            #     file_name = 'signal_based_bearing_defect_model_inner.npy'
            #     path = os.path.join(save_path, file_name)
            #     np.save(path, np.array(acc))
            # elif output_file == 3:
            #     file_name = "signal_based_bearing_defect_model_inner.csv"
            #     dataframe = pd.DataFrame(acc)
            #     path = os.path.join(save_path, file_name)
            #     dataframe.to_csv(path, index=False, sep=',')
            # elif output_file == 4:
            #     file_name = "signal_based_bearing_defect_model_inner.txt"
            #     path = os.path.join(save_path, file_name)
            #     np.savetxt(path, acc)

        # else:
        #     if output_image == 0:
        #         file_name = "%s.png" % "signal_based_bearing_defect_model"
        #         path = os.path.join(save_path, file_name)
        #         plt.savefig(path)
        #     elif output_image == 1:
        #         file_name1 = "%s.png" % "signal_based_bearing_defect_model"
        #         file_name2 = "%s.jpg" % "signal_based_bearing_defect_model"
        #         path1 = os.path.join(save_path, file_name1)
        #         path2 = os.path.join(save_path, file_name2)
        #         plt.savefig(path1)
        #         plt.savefig(path2)
        #     elif output_image == 2:
        #         file_name1 = "%s.png" % "signal_based_bearing_defect_model"
        #         file_name2 = "%s.svg" % "signal_based_bearing_defect_model"
        #         path1 = os.path.join(save_path, file_name1)
        #         path2 = os.path.join(save_path, file_name2)
        #         plt.savefig(path1)
        #         plt.savefig(path2)
        #     elif output_image == 3:
        #         file_name1 = "%s.png" % "signal_based_bearing_defect_model"
        #         file_name2 = "%s.pdf" % "signal_based_bearing_defect_model"
        #         path1 = os.path.join(save_path, file_name1)
        #         path2 = os.path.join(save_path, file_name2)
        #         plt.savefig(path1)
        #         plt.savefig(path2)
        #     plt.close()

            # if output_file == 0:
            #     file_name = 'signal_based_bearing_defect_model_ball.mat'
            #     path = os.path.join(save_path, file_name)
            #     sio.savemat(path, {'data': acc})
            # elif output_file == 1:
            #     dataframe = pd.DataFrame(acc)
            #     file_name = 'signal_based_bearing_defect_model_ball.xlsx'
            #     path = os.path.join(save_path, file_name)
            #     writer = pd.ExcelWriter(path)
            #     dataframe.to_excel(writer)
            #     writer.save()
            # elif output_file == 2:
            #     file_name = 'signal_based_bearing_defect_model_ball.npy'
            #     path = os.path.join(save_path, file_name)
            #     np.save(path, np.array(acc))
            # elif output_file == 3:
            #     file_name = "signal_based_bearing_defect_model_ball.csv"
            #     dataframe = pd.DataFrame(acc)
            #     path = os.path.join(save_path, file_name)
            #     dataframe.to_csv(path, index=False, sep=',')
            # elif output_file == 4:
            #     file_name = "signal_based_bearing_defect_model_ball.txt"
            #     path = os.path.join(save_path, file_name)
            #     np.savetxt(path, acc)
        # else:
        # plt.show()
    print(time, acc)

    return time, acc


if __name__ == '__main__':
    pass
