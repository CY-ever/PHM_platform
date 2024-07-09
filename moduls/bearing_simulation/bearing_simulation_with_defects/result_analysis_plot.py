## the following code was designed for result analysis and plots

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import hilbert, detrend
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import pearsonr
from scipy import signal
from moduls.bearing_simulation.bearing_simulation_with_defects.GlobalVariable import globalVariables as gbv


# @function
def envelope_spectrum(signal_data, sampling_frequency, *args, **kwargs):
    """
    1) this function obtains the signal envelope spectrum by Hilbert transform and FFT;
    2) also, the signal has been detrended (both "constant" and "linear") before envelope spectrum extraction
    """
    signal_length = len(signal_data)
    n_data_num = np.arange(signal_length)
    delta_T = signal_length / sampling_frequency
    frequency_x = n_data_num / delta_T
    signal_data = detrend(signal_data, type='constant')
    signal_data = detrend(signal_data, type='linear')
    signal_hilbert = np.abs(hilbert(signal_data))
    signal_hilbert = detrend(signal_hilbert, type='constant')
    signal_hilbert = detrend(signal_hilbert, type='linear')
    envelope_spectrum_y = np.abs(fft(signal_hilbert) / (signal_length / 2))
    envelope_spectrum_y[0] = envelope_spectrum_y[0] / 2
    envelope_spectrum_y[int(signal_length / 2) - 1] = envelope_spectrum_y[int(signal_length / 2) - 1] / 2
    frequency_x = frequency_x[0:int(signal_length / 2) - 1]
    envelope_spectrum_y = envelope_spectrum_y[0:int(signal_length / 2) - 1]
    return frequency_x, envelope_spectrum_y


def result_analysis_plot(time=None, acc_x=None, acc_y=None, name_parameter=None,
                         outer_ring_switch=None, inner_ring_switch=None, ball_switch=None,
                         save_path="", output_image=0, *args, **kwargs):
    # varargin = result_analysis_plot.varargin
    # nargin = result_analysis_plot.nargin

    sim_parameter = gbv.sim_parameter

    fs = 1 / sim_parameter.step_size

    ## the 1st figure is to present the acc in time domain;
    # plt.figure(dpi=800)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, acc_x, 'r')
    # plt.xlim(0.4, 0.8)
    # plt.ylim(-0.001, 0.001)
    plt.xlabel('time [s]')
    plt.ylabel('acc_x [m·s^-2]')
    plt.title('x-acceleration')
    # subplot(2,2,2)
    # plot(time,acc_y,'r');
    # xlabel('time(s)');
    # ylabel('acc\_y (m/s^2)');
    # title('y-acceleration');
    plt.subplots_adjust(wspace=1, hspace=0.7)

    ## the 2nd figure is to presnet the acc in frequency domain, and compare with theoretical BPFs.
    plt.subplot(2, 1, 2)
    # pEnvOuter_x, fEnvOuter_x, __, __ = envspectrum(acc_x, fs, nargout=4)
    # pEnvOuter_x, fEnvOuter_x, __, __ = envelope_spectrum(acc_x, fs, nargout=4)
    fEnvOuter_x, pEnvOuter_x = envelope_spectrum(acc_x, fs)

    plt.plot(fEnvOuter_x, pEnvOuter_x)
    # plt.plot(pEnvOuter_x, fEnvOuter_x)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('amplitude')
    plt.title('envelope spectrum of acc_x')
    file_name = name_parameter.figure_name

    if outer_ring_switch == 1:
        if output_image == 0:
            file_name = f"time and frequency domain acceleration0.png"
            path = os.path.join(save_path, file_name)
            plt.savefig(path)
        elif output_image == 1:
            file_name1 = f"time and frequency domain acceleration0.png"
            file_name2 = f"time and frequency domain acceleration0.jpg"
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        elif output_image == 2:
            file_name1 = f"time and frequency domain acceleration0.png"
            file_name2 = f"time and frequency domain acceleration0.svg"
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        elif output_image == 3:
            file_name1 = f"time and frequency domain acceleration0.png"
            file_name2 = f"time and frequency domain acceleration0.pdf"
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        plt.close()

    if inner_ring_switch == 1:
        if output_image == 0:
            file_name = f"time and frequency domain acceleration1.png"
            path = os.path.join(save_path, file_name)
            plt.savefig(path)
        elif output_image == 1:
            file_name1 = f"time and frequency domain acceleration1.png"
            file_name2 = f"time and frequency domain acceleration1.jpg"
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        elif output_image == 2:
            file_name1 = f"time and frequency domain acceleration1.png"
            file_name2 = f"time and frequency domain acceleration1.svg"
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        elif output_image == 3:
            file_name1 = f"time and frequency domain acceleration1.png"
            file_name2 = f"time and frequency domain acceleration1.pdf"
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        plt.close()

    if ball_switch == 1:
        if output_image == 0:
            file_name = f"time and frequency domain acceleration2.png"
            path = os.path.join(save_path, file_name)
            plt.savefig(path)
        elif output_image == 1:
            file_name1 = f"time and frequency domain acceleration2.png"
            file_name2 = f"time and frequency domain acceleration2.jpg"
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        elif output_image == 2:
            file_name1 = f"time and frequency domain acceleration2.png"
            file_name2 = f"time and frequency domain acceleration2.svg"
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        elif output_image == 3:
            file_name1 = f"time and frequency domain acceleration2.png"
            file_name2 = f"time and frequency domain acceleration2.pdf"
            path1 = os.path.join(save_path, file_name1)
            path2 = os.path.join(save_path, file_name2)
            plt.savefig(path1)
            plt.savefig(path2)
        plt.close()

    # plt.savefig(name_temp)
    # plt.show()
    return 0


if __name__ == '__main__':
    pass
