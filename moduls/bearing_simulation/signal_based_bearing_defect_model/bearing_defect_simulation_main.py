# this script was designed to run the whole simulation
# author: Diwang Ruan
# date: 2020.09.18
# version: V1.0

import numpy as np
# from . import bearing_characteristic_frequency as bcf
# from . import signal_based_bearing_defect_model as sbbdm
import moduls.bearing_simulation.signal_based_bearing_defect_model.bearing_characteristic_frequency as bcf
import moduls.bearing_simulation.signal_based_bearing_defect_model.signal_based_bearing_defect_model as sbbdm
from moduls.bearing_simulation.signal_segmentaion import signal_segmentaion
from flask import abort
import os
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from utils.save_data import data_save_for_2
from moduls.bearing_simulation.Report_SCR import word_signal


# from flask import abort


class BearingParameter:
    """
    :param
     D: ball diameter£¬-mm;
     di: inner ring raceway contact diameter£¬-mm;
     do: outer ring raceway contact diameter£¬-mm;
     d: pitch diameter, -mm£»;
     contact_angle: initial contact angle, -deg;
     Z: number of balls;
     bearing_type_factor: bearing type parameter, 3/2 for ball bearings, and 10/9 for roller bearings;
    """
    D = 22.225
    di = 102.7938
    do = 147.7264
    d = np.dot(0.5, (di + do))
    contact_angle = 40
    Z = 16
    bearing_type_factor = 3 / 2


class ConditionParameter:
    """
    :param
    load_max: maximum of external radial load, N;
    load_proportional_factor: modification coefficient of load;
    shaft_speed: shaft speed, -Hz;
    resonance_frequency: resonance frequency of bearing, -Hz;
    phi_limit: extent of load zone, deg; (0-90]
    load_distribution_parameter: load distribution parameter, (0-0.5)

    """
    load_max = 1000
    load_proportional_factor = 0.1
    shaft_speed = 25
    resonance_frequency = 3000
    phi_limit = 80
    load_distribution_parameter = 0.3


class DefectParameter:
    """
    :param
    defect_type: 1: outer ring defect; 2: inner ring defect; 3: ball defect;
    defect_frequency: BPFO or BPFI or BSF, depends on defined defect type;
    decaying_parameter: the decay parameter B in exponential decaying function,
                        larger value brings faster decay rate.
    defect_initial_position: initial angular position of defect,-deg;
    """

    defect_type = 1
    defect_frequency = None
    decaying_parameter = 300
    defect_initial_position = 15


class SimParameter:
    """
    :param
    step_size: simulation step_size, s;
    duration: simulation duration, s;
    """
    step_size = 0.0001
    duration = 1


def signal_one_main(D=22.225, di=102.7938, do=147.7264, Z=16, contact_angle=40, bearing_type_factor=3 / 2,
                    load_max=1000, load_proportional_factor=0.1,
                    shaft_speed=25, resonance_frequency=3000, phi_limit=80, load_distribution_parameter=0.3,
                    defect_type=1, decaying_parameter=300, defect_initial_position=15,
                    step_size=0.0001, duration=1,
                    save_path=None, output_file=0,
                    output_image=0):
    """
    :param: D: ball diameter£¬-mm;
    :param: di:inner ring raceway contact diameter£¬-mm;
    :param: do: outer ring raceway contact diameter£¬-mm;
    :param: d: pitch diameter, -mm£»;
    :param: contact_angle: initial contact angle, -deg;
    :param: Z: number of balls;
    :param: bearing_type_factor: bearing type parameter, 3/2 for ball bearings, and 10/9 for roller bearings;
    :param: load_max: maximum of external radial load, N;
    :param: load_proportional_factor: modification coefficient of load;
    :param: shaft_speed: shaft speed, -Hz;
    :param: resonance_frequency: resonance frequency of bearing, -Hz;
    :param: phi_limit: extent of load zone, deg; (0-90]
    :param: load_distribution_parameter: load distribution parameter, (0-0.5)
    :param: defect_type: 0: outer ring defect; 1: inner ring defect; 2: ball defect;
    :param: defect_frequency: BPFO or BPFI or BSF, depends on defined defect type;
    :param: decaying_parameter: the decay parameter B in exponential decaying function,
                        larger value brings faster decay rate.
    :param: defect_initial_position: initial angular position of defect,-deg;
    :param: step_size: simulation step_size, s;
    :param: duration: simulation duration, s;

    :return: acc 加速度信号，纵坐标
    :return: time 时间轴
    """

    bearing_parameter = BearingParameter()
    bearing_parameter.D = D
    bearing_parameter.di = di
    bearing_parameter.do = do
    bearing_parameter.d = np.dot(0.5, (di + do))
    bearing_parameter.contact_angle = contact_angle
    bearing_parameter.Z = Z
    bearing_parameter.bearing_type_factor = bearing_type_factor

    # return bearing_parameter

    # condition parameter structure
    # condition_parameter = struct('load_max', [], 'load_proportional_factor', [], 'shaft_speed', [],
    #                              'resonance_frequency',
    #                              [], 'phi_limit', [], 'load_distribution_parameter', [])
    condition_parameter = ConditionParameter()

    condition_parameter.load_max = load_max
    condition_parameter.load_proportional_factor = load_proportional_factor
    condition_parameter.shaft_speed = shaft_speed
    condition_parameter.resonance_frequency = resonance_frequency
    condition_parameter.phi_limit = phi_limit
    condition_parameter.load_distribution_parameter = load_distribution_parameter

    # # defect parameter structure
    # defect_parameter = struct('defect_type', [], 'defect_frequency', [], 'decaying_parameter', [],
    #                           'defect_initial_position', [])
    defect_parameter = DefectParameter()
    defect_parameter.defect_type = defect_type
    defect_parameter.defect_frequency = bcf.bearing_characteristic_frequency(
        [bearing_parameter.D, bearing_parameter.d, bearing_parameter.contact_angle, bearing_parameter.Z,
         condition_parameter.shaft_speed], defect_parameter.defect_type)

    defect_parameter.decaying_parameter = decaying_parameter
    defect_parameter.defect_initial_position = defect_initial_position

    # simulation parameter strucure
    # sim_parameter = struct('step_size', [], 'duration', [])
    sim_parameter = SimParameter()
    sim_parameter.step_size = step_size
    sim_parameter.duration = duration

    # demostration on how to call "signal_based_bearing_defect_model"
    time_data, acc = sbbdm.signal_based_bearing_defect_model(bearing_parameter, defect_parameter, condition_parameter,
                                                             sim_parameter,
                                                             save_path=save_path,
                                                             output_file=output_file,
                                                             output_image=output_image)
    label = np.zeros(shape=(1,), dtype=np.int)
    label = label + defect_type
    # print("label", label)
    # return time, acc,
    return acc, label


def signal_main(outer_sig=1, inner_sig=0, ball_sig=0, D=22.225, di=102.7938, do=147.7264, Z=16, contact_angle=40,
                bearing_type_factor=3 / 2,
                load_max=1000, load_proportional_factor=0.1,
                shaft_speed=25, resonance_frequency=3000, phi_limit=80, load_distribution_parameter=0.3,
                decaying_parameter=300, defect_initial_position=15,
                step_size=0.0001, duration=1, length=5000, shift=2000, segmentation_sig=0,
                save_path="./", output_file=0, output_image=0):
    # from flask import abort
    # abort(400, "abc")

    if D < 1 or D > 440:
        abort(400, "ERROR: The allowable range of D value is between 1 and 440 millimeters!!!")

    if di < 1 or di > 2000:
        abort(400, "ERROR: The allowable range of Di value is between 1 and 2000 millimeters!!!")

    if do < 1 or do > 2500:
        abort(400, "ERROR: The allowable range of Do value is between 1 and 2500 millimeters!!!")

    if Z < 8 or Z > 30:
        abort(400, "ERROR: The range of Z value is abnormal!!!")

    if contact_angle < 0 or contact_angle > 60:
        abort(400, "ERROR: The range of α value is abnormal!!!")

    if bearing_type_factor < 0:
        abort(400, "ERROR: The value of Bearing type factor must be greater than 0!!!")

    if load_max < 0:
        abort(400, "ERROR: The value of Load max must be greater than 0!!!")

    if load_proportional_factor < 0:
        abort(400, "ERROR: The value of Load proportional factor must be greater than 0!!!")

    if shaft_speed < 0:
        abort(400, "ERROR: The value of Shaft speed must be greater than 0!!!")

    if resonance_frequency < 0:
        abort(400, "ERROR: The value of Resonance frequency must be greater than 0!!!")

    if phi_limit <= 0 or phi_limit > 90:
        abort(400, "ERROR: The allowable range of φ limit value is (0,90] !!!")

    if load_distribution_parameter <= 0 or load_distribution_parameter >= 0.5:
        abort(400, "ERROR: The allowable range of Load distribution value is (0,0.5)!!!")

    if decaying_parameter < 0:
        abort(400, "ERROR: The value of B must be greater than 0!!!")

    if defect_initial_position < 0 or defect_initial_position > 360:
        abort(400, "ERROR: The allowable range of Defect initial position value is between 0 and 360 degree!!!")

    if step_size < 0 or step_size > 1:
        abort(400, "ERROR: The allowable range of Step size value is between 0 and 1!!!")

    if duration < 0:
        abort(400, "ERROR: The value of Duration must be greater than 0!!!")
    # if D < 1 or D > 440:
    #     abort(400, "The allowable range of D value is between 1 and 440 millimeters!!!")
    # if D < 1 or D > 440:
    #     abort(400, "The allowable range of D value is between 1 and 440 millimeters!!!")

    defect_type_list = []

    if outer_sig:
        defect_type_list.append(0)
    if inner_sig:
        defect_type_list.append(1)
    if ball_sig:
        defect_type_list.append(2)

    all_data = None
    all_label = None
    all_data_org = None
    for defect_type in defect_type_list:
        if defect_type in [0, 1, 2]:
            time_data, label = signal_one_main(D=D, di=di, do=do, Z=Z, contact_angle=contact_angle,
                                               bearing_type_factor=bearing_type_factor,
                                               load_max=load_max,
                                               load_proportional_factor=load_proportional_factor,
                                               shaft_speed=shaft_speed,
                                               resonance_frequency=resonance_frequency,
                                               phi_limit=phi_limit,
                                               load_distribution_parameter=load_distribution_parameter,
                                               defect_type=defect_type,
                                               decaying_parameter=decaying_parameter,
                                               defect_initial_position=defect_initial_position,
                                               step_size=step_size, duration=duration,
                                               save_path=save_path,
                                               output_file=output_file, output_image=output_image)
            # print("label:", label)

            one_date = time_data
            if segmentation_sig:
                time_data = np.reshape(time_data, (1, -1))
                time_data = signal_segmentaion(time_data, length, shift)
                N = time_data.shape[0]
                label = label * np.ones(N, dtype=int)
                label = np.reshape(label, (N, 1))




        # else:
        #     time_data = [0.1 for i in range(10000)]
        #     label = np.zeros(shape=(1,), dtype=np.int)
        #     print("label:", label)
        #     label = label + defect_type
        #     print("label:", label)

        if all_data is None:
            all_data = time_data
            all_label = label
            all_data_org = one_date
        else:
            all_data = np.vstack((all_data, time_data))
            all_label = np.vstack((all_label, label))
            all_data_org = np.vstack((all_data_org, one_date))


    all_label = all_label.astype(int)

    if len(all_data.shape) == 1:
        all_data = all_data.reshape((1, -1))
    if len(all_label.shape) == 1:
        all_label = all_label.reshape((1, -1))
    print("all_data", all_data.shape)
    print("all_label", all_label.shape)

    a = []
    if outer_sig == 1:
        a.append(0)
    if inner_sig == 1:
        a.append(1)
    if ball_sig == 1:
        a.append(2)

    if len(all_data_org.shape) == 1:
        all_data_org = all_data_org.reshape((1, -1))


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(save_path):
        for i in range(len(a)):
            title_string = ['outer ring', 'inner ring', 'ball']
            title_temp = 'Acceleration of bearing with defect on ' + title_string[a[i]]

            plt.plot(np.arange(0, duration, step_size), all_data_org[i, :])
            plt.title(title_temp)
            plt.xlabel('Time [s]')
            plt.ylabel('Acceleration [m·s^-2]')
            if output_image == 0:
                file_name = "%s.png" % ("signal_based_bearing_defect_model" + str(i+1))
                path = os.path.join(save_path, file_name)
                plt.savefig(path)
            elif output_image == 1:
                file_name1 = "%s.png" % ("signal_based_bearing_defect_model" + str(i+1))
                file_name2 = "%s.jpg" % ("signal_based_bearing_defect_model" + str(i+1))
                path1 = os.path.join(save_path, file_name1)
                path2 = os.path.join(save_path, file_name2)
                plt.savefig(path1)
                plt.savefig(path2)
            elif output_image == 2:
                file_name1 = "%s.png" % ("signal_based_bearing_defect_model" + str(i+1))
                file_name2 = "%s.svg" % ("signal_based_bearing_defect_model" + str(i+1))
                path1 = os.path.join(save_path, file_name1)
                path2 = os.path.join(save_path, file_name2)
                plt.savefig(path1)
                plt.savefig(path2)
            elif output_image == 3:
                file_name1 = "%s.png" % ("signal_based_bearing_defect_model" + str(i+1))
                file_name2 = "%s.pdf" % ("signal_based_bearing_defect_model" + str(i+1))
                path1 = os.path.join(save_path, file_name1)
                path2 = os.path.join(save_path, file_name2)
                plt.savefig(path1)
                plt.savefig(path2)
            plt.close()

        data_save_for_2(all_data, all_label, output_file, save_path,
                        file_name="signal_based_bearing_defect_model",
                        file_name1="signal_based_bearing_defect_model_label",
                        index_label1="(sample_size, sample_length)")

    # report
    word_signal(save_path=save_path, output_file=output_file, output_image=output_image,
                all_data=all_data, all_label=all_label,
                D=D, di=di, do=do, Z=Z, α=contact_angle, bearing_type_factor=bearing_type_factor,
                load_max=load_max,
                load_proportional_factor=load_proportional_factor,
                shaft_speed=shaft_speed,
                resonance_frequency=resonance_frequency,
                phi_limit=phi_limit,
                load_distribution_parameter=load_distribution_parameter,
                B=decaying_parameter,
                defect_initial_position=defect_initial_position,
                outer_sig=outer_sig, inner_sig=inner_sig, ball_sig=ball_sig,
                step_size=step_size, duration=duration,
                length=length, shift=shift, segmentation_sig=segmentation_sig
                )

    return all_data, all_label


if __name__ == '__main__':
    all_data, all_label = signal_main()
    # all_data, all_label = signal_main(defect_type_list=(1, 2, ))
    """
    defect_type_list=(0, 1, 2)
    output:
    (3, 10000)
    (3, )"""

# if __name__ == '__main__':
#     time, acc = signal_main(defect_type=1)
