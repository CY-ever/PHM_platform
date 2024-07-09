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
from . import amplitude as am
from . import exponential_decay
import os


def signal_based_bearing_defect_model(bearing_parameter=None, defect_parameter=None, condition_parameter=None,
                                      sim_parameter=None, save_path=None):
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
        Am = am.amplitude(sim_time, bearing_parameter, condition_parameter, defect_parameter)

        x_temp = 0

        for k in np.arange(1, num_impulse).reshape(-1):
            x = exponential_decay.exponential_decay(sim_time, k, defect_parameter, condition_parameter)
            x_temp = x_temp + np.dot(Am, x)

        x_out[int(sim_time / step_size)] = x_temp

    time = np.arange(0, duration, step_size)

    acc = np.copy(x_out)

    # plot for time-acc
    title_string = ['outer ring', 'inner ring', 'ball']

    title_temp = ['acceleration of bearing with defect on', title_string[defect_parameter.defect_type]]

    plt.plot(np.arange(0, duration, step_size), x_out)
    plt.title(title_temp)
    # plt.title('title')
    plt.xlabel('time')
    plt.ylabel('acceleration')

    save_path = "signal_based_figure"

    # name_list = ["outer", "inner", "ball"]
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if defect_parameter.defect_type == 0:
            plt.savefig(save_path+"/outer.png")
        elif defect_parameter.defect_type == 1:
            plt.savefig(save_path+'/inner.png')
        else:
            plt.savefig(save_path+'/ball.png')

    # else:
    #     plt.show()

    return time, acc


if __name__ == '__main__':
    pass
