# this script was designed to run the whole simulation
# author: Diwang Ruan
# date: 2020.09.18
# version: V1.0

import numpy as np
from . import bearing_characteristic_frequency as bcf
from . import signal_based_bearing_defect_model as sbbdm


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


def signal_main(D=22.225, di=102.7938, do=147.7264, Z=16, contact_angle=40, bearing_type_factor=3 / 2,
                load_max=1000, load_proportional_factor=0.1,
                shaft_speed=25, resonance_frequency=3000, phi_limit=80, load_distribution_parameter=0.3,
                defect_type=1, decaying_parameter=300, defect_initial_position=15,
                step_size=0.0001, duration=1,
                save_path=None):
    """

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
    time, acc = sbbdm.signal_based_bearing_defect_model(bearing_parameter, defect_parameter, condition_parameter,
                                                        sim_parameter, save_path=save_path)

    return time, acc


if __name__ == '__main__':
    import pandas as pd

    time, acc = signal_main()
    print(acc)
    print(time)

