## this code was developed to generate the exponential decay function
# author: Diwang Ruan
# date: 2020.09.17
# version: V1.0

## introduction of "exponential_decay"
# function: to simulate the decaying apmplitude of a unit impulse at any time;
# input:
# sim_time: current simulation clock, -s;
# impulse_flag: serial number of impulse, like No.x impulse;
# defect_parameter: structure for defect definition, defined in "bearing_defect_simulation.m"
# condition_parameter: structure for condition definition, defined in "bearing_defect_simulation.m"
# output:
# decaying_signal: the amplitude of decaying impulse at current time, a signle value at every sim_time.

## the main code of "exponential_decay".

import numpy as np


def exponential_decay(sim_time=None, impulse_flag=None, defect_parameter=None, condition_parameter=None):
    # varargin = exponential_decay.varargin
    # nargin = exponential_decay.nargin

    # assignment for main parameters
    B = defect_parameter.decaying_parameter
    # exponential_decay.m:22

    fn = condition_parameter.resonance_frequency
    # exponential_decay.m:23

    defect_frequency = defect_parameter.defect_frequency
    # exponential_decay.m:24

    start_time = impulse_flag / defect_frequency
    # exponential_decay.m:25

    # the piece-wise function of exponential decay process
    if sim_time < start_time:
        decaying_signal = 0
    # exponential_decay.m:29
    else:
        decaying_signal = np.dot(np.exp(np.dot(- B, (sim_time - start_time))),
                                 np.cos(np.dot(np.dot(np.dot(2, np.pi), fn), sim_time)))
    # exponential_decay.m:31

    return decaying_signal


if __name__ == '__main__':
    pass
