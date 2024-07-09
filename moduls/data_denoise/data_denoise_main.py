from moduls.data_denoise.DWT.DWT_Filter import DWT_function
from moduls.data_denoise.DWT.DWT_Fs import DWT_Fs_function
from moduls.data_denoise.DWT.DWT_kurtosis_max import DWT_kurtosis_max_function
from moduls.data_denoise.DWT.DWT_threshold_filter import DWT_threshold_filter
from moduls.data_denoise.EMD.EMD_main import filter_with_emd_main
from moduls.data_denoise.fast_kurtogram.Fast_kurtogram_main import FK_filterdesign
from moduls.data_denoise.FFT.FFT_Filter_main import FFT_filter
from moduls.data_denoise.mean.Mean_Filter_main import mean_denoise_main
from moduls.data_denoise.physics_based_PSD_filter_threshold.physics_based_PSD_filter_threshold_main import PSD_filter
from moduls.data_denoise.WaveletPacket.WaveletPacket_Filter import WaveletPacket_Filter
from moduls.data_denoise.WaveletPacket.WaveletPacket_Fs import WaveletPacket_Fs
from moduls.data_denoise.WaveletPacket.WaveletPacket_kurtosis_max import WaveletPacket_kurtosis_max
from moduls.data_denoise.EMD.writein import writein
from moduls.data_denoise.Report_signal_denoise import *
from flask import abort
import numpy as np

from utils.table_setting import *
from utils.save_data import save_data_func


def data_denoise_main(signal, switch=7, DWT_select=3, DWT_Filter_name='db1', DWT_Filter_N=3, DWT_Fs_name='db1',
                      DWT_Fs_F=10000, DWT_Fs_Fs=(3000, 4500), DWT_kurtosis_max_name='db1', DWT_kurtosis_max_N=3,
                      DWT_kurtosis_max_k=3, DWT_threshold_name='db1', DWT_threshold_max_N=3, threshold_method=0,
                      threshold_coeff=0.5, EMD_N=4, FK_nlevel=6, FK_Fs=12000, FK_mode='lowpass', FK_order=8,
                      Kurtosis_figure=True, FFT_critical_freqs=(0, 2000), FFT_mode='lowpass', Mean_filt_length=10,
                      PSD_filter_fr=37.5, PSD_filter_n_ball=8, PSD_filter_d_ball=7.92, PSD_filter_d_pitch=34.55,
                      PSD_filter_alpha=0, PSD_filter_frequency_band_max=2000, PSD_filter_factor=0.5,
                      PSD_filter_sideband_switch=0, PSD_filter_sampling_frequency=25600,
                      PSD_filter_cut_off_frequency=1000, PSD_filter_filter_method=1, PSD_filter_filter_order=6,
                      WaveletPacket_select=0, WaveletPacket_Filter_name='db1', WaveletPacket_Filter_N=5,
                      WaveletPacket_Fs_name='db1', WaveletPacket_Fs_sampling_frequency=10000,
                      WaveletPacket_Fs_band=(3000, 4500), WaveletPacket_kurtosis_max_name='db1',
                      WaveletPacket_kurtosis_max_N=3, WaveletPacket_kurtosis_max_k=2, save_path='./', output_file=0,
                      output_image=0):
    '''
    :param signal:输入可以是二维数组也可以是一维数组
    :param return: 输出一律为二维，如（1，10001），（5，10001）
    '''
    global data
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)

    m = signal.shape[0]
    data_all = []
    for i in range(m):
        signal1 = signal[i]  # 一维

        if switch == 2:
            if DWT_select == 0:
                data = DWT_function(signal1, DWT_Filter_name, DWT_Filter_N, save_path, output_file, output_image)

            elif DWT_select == 2:

                data = DWT_Fs_function(signal1, DWT_Fs_name, DWT_Fs_F, DWT_Fs_Fs, save_path, output_file, output_image)

            elif DWT_select == 1:

                data = DWT_kurtosis_max_function(signal1, DWT_kurtosis_max_name, DWT_kurtosis_max_N, DWT_kurtosis_max_k,
                                                 save_path, output_file, output_image)

            elif DWT_select == 3:

                data = DWT_threshold_filter(signal1, DWT_threshold_name, DWT_threshold_max_N, threshold_method,
                                            threshold_coeff,
                                            save_path, output_file, output_image)

        elif switch == 3:

            data = filter_with_emd_main(signal1, EMD_N, save_path, output_file, output_image)

        elif switch == 6:
            data = FK_filterdesign(signal1, FK_nlevel, FK_Fs, FK_mode, FK_order, Kurtosis_figure, save_path,
                                   output_file, output_image)
        elif switch == 1:
            data = FFT_filter(signal1, FFT_critical_freqs, FFT_mode, save_path, output_file, output_image)
        elif switch == 4:

            data = mean_denoise_main(signal1, Mean_filt_length, save_path, output_file, output_image)

        elif switch == 7:

            data = PSD_filter(signal1, PSD_filter_fr, PSD_filter_n_ball, PSD_filter_d_ball, PSD_filter_d_pitch,
                              PSD_filter_alpha, PSD_filter_frequency_band_max, PSD_filter_factor,
                              PSD_filter_sideband_switch, PSD_filter_sampling_frequency, PSD_filter_cut_off_frequency,
                              PSD_filter_filter_method, PSD_filter_filter_order, save_path, output_file, output_image)

        elif switch == 5:
            if WaveletPacket_select == 0:

                data = WaveletPacket_Filter(signal1, WaveletPacket_Filter_name, WaveletPacket_Filter_N, save_path,
                                            output_file, output_image)

            elif WaveletPacket_select == 2:

                data = WaveletPacket_Fs(signal1, WaveletPacket_Fs_name, WaveletPacket_Fs_sampling_frequency,
                                        WaveletPacket_Fs_band, save_path, output_file, output_image)

            elif WaveletPacket_select == 1:

                data = WaveletPacket_kurtosis_max(signal1, WaveletPacket_kurtosis_max_name,
                                                  WaveletPacket_kurtosis_max_N, WaveletPacket_kurtosis_max_k, save_path,
                                                  output_file, output_image)

        data_all.append(data)
        data_denoise_all = np.array(data_all)  # 二维

    # 生成报告
    if switch == 1:
        word_FFT(inputdata=signal,
                 outputdata=data_denoise_all,
                 critical_freqs=FFT_critical_freqs,
                 mode=FFT_mode, output_file=output_file,
                 output_image=output_image, save_path=save_path)
    elif switch == 2:
        if DWT_select == 0:
            word_DWT_simple_filter(inputdata=signal, outputdata=data_denoise_all, name=DWT_Filter_name, N=DWT_Filter_N,
                                   output_file=output_file, output_image=output_image, save_path=save_path)
        elif DWT_select == 1:
            word_DWT_kurtosis_max(inputdata=signal, outputdata=data_denoise_all, name=DWT_kurtosis_max_name,
                                  N=DWT_kurtosis_max_N, k=DWT_kurtosis_max_k, output_file=output_file,
                                  output_image=output_image, save_path=save_path)
        elif DWT_select == 2:
            word_DWT_Fs(inputdata=signal, outputdata=data_denoise_all, name=DWT_Fs_name, F=DWT_Fs_F, Fs=DWT_Fs_Fs,
                        output_file=output_file,
                        output_image=output_image, save_path=save_path)
        else:
            word_DWT_threshold(inputdata=signal, outputdata=data_denoise_all, name=DWT_threshold_name,
                               N=DWT_threshold_max_N, threshold_method=threshold_method, output_file=output_file,
                               output_image=output_image, threshold_coeff=threshold_coeff, save_path=save_path)
    elif switch == 3:
        word_EMD(inputdata=signal, outputdata=data_denoise_all, N=EMD_N, output_file=output_file,
                 output_image=output_image,
                 save_path=save_path)
    elif switch == 4:
        word_mean(inputdata=signal, outputdata=data_denoise_all, filt_length=Mean_filt_length, output_file=output_file,
                  output_image=output_image, save_path=save_path)
    elif switch == 5:
        if WaveletPacket_select == 0:
            word_WP_simple_filter(inputdata=signal, outputdata=data_denoise_all, name=WaveletPacket_Filter_name,
                                  N=WaveletPacket_Filter_N, output_file=output_file, output_image=output_image,
                                  save_path=save_path)
        elif WaveletPacket_select == 1:
            word_WP_kurtosis_max(inputdata=signal, outputdata=data_denoise_all, name=WaveletPacket_kurtosis_max_name,
                                 N=WaveletPacket_kurtosis_max_N, k=WaveletPacket_kurtosis_max_k,
                                 output_file=output_file, output_image=output_image, save_path=save_path)
        else:
            word_WP_Fs(inputdata=signal, outputdata=data_denoise_all, name=WaveletPacket_Fs_name,
                       sampling_frequency=WaveletPacket_Fs_sampling_frequency,
                       Fs_band=WaveletPacket_Fs_band, output_file=output_file, output_image=output_image,
                       save_path=save_path)
    elif switch == 6:
        word_FK(inputdata=signal, outputdata=data_denoise_all, nlevel=FK_nlevel, Fs=FK_Fs, mode=FK_mode, order=FK_order,
                Kurtosis_figure=Kurtosis_figure, output_file=output_file, output_image=output_image,
                save_path=save_path)
    elif switch == 7:
        word_PSD(inputdata=signal, outputdata=data_denoise_all, fr=PSD_filter_fr, n_ball=PSD_filter_n_ball,
                 d_ball=PSD_filter_d_ball,
                 d_pitch=PSD_filter_d_pitch, alpha=PSD_filter_alpha, frequency_band_max=PSD_filter_frequency_band_max,
                 factor=PSD_filter_factor,
                 sideband_switch=PSD_filter_sideband_switch, sampling_frequency=PSD_filter_sampling_frequency,
                 cut_off_frequency=PSD_filter_cut_off_frequency, filter_method=PSD_filter_filter_method,
                 filter_order=PSD_filter_filter_order, output_file=output_file, output_image=output_image,
                 save_path=save_path)
    return data_denoise_all


if __name__ == "__main__":
    signal = writein('1.mat', 1)  # 二维
    # print(signal.ndim)
    x = signal[:2, :]
    a = data_denoise_main(x, switch=6, Kurtosis_figure=True, save_path="./result")
    print(a.shape)  # 输出为二维
