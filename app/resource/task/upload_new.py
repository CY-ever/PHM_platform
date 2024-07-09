import os
import shutil
import pandas as pd
import numpy as np
from flask import g, request, abort
from flask import Flask, send_from_directory
from flask_restful import Resource
from flask_restful.reqparse import RequestParser
from sqlalchemy.orm import load_only

from app import db
from model.user import UserParameter
from moduls.data_augmentation.Monte_Carlo_sampling.Monte_Carlo_main import Monte_Carlo_DA
from moduls.ml import report
from utils import parser
from utils.decorators import login_required
from utils.save_data import save_data_func


class StartTasks(Resource):
    """开始用户任务类视图"""
    # 添加装饰器
    method_decorators = {
        "post": [login_required]
    }
    """
        # 1.获取参数
        # 1.1 数据文件
        # 1.2 user_id通过g对象获取
        # 2.校验参数
        # 3.逻辑处理
        # 3.1 对方法列表进行拆包
        # 3.2 根据制定法功法调取制定函数方法
        # 4.返回值:传递工作完成信息
    """

    def post(self):
        # 1.获取参数
        # 2.校验参数
        # 2.1 模块参数
        print(request.json)

        rp = RequestParser()
        rp.add_argument("bearing_simulation", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("data_augmentation", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("data_preprocessing", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("features_extraction", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("fault_diagnose", type=parser.modul_parameter, required=True, location="json")

        rp.add_argument("output_files", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("output_images", type=parser.modul_parameter, required=True, location="json")

        # 2.2 模块一参数
        rp.add_argument("Do_phy", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Di_phy", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("d_phy", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Kb_phy", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("alpha_phy", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Nb", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Ms", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Mp", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Mr", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Ks", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Kp", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Kr", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Cs", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Cp", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Cr", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("L", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("B_phy", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("H", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("ORS", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("ORN", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("OR_defect_position", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("IRS", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("IRN", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("IR_defect_position", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("BS", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("BN", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("B_defect_position", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("B_defect_identifier", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Fr", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Fa", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Omega_shaft", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("step_size_phy", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("duration_phy", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("mutation_percentage", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("initial_angular_position", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Do_sig", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Di_sig", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("z_sig", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("D_sig", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("type_factor_sig", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("alpha_sig", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("load_max", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("shaft_speed", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("phi_limit", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("load_proportional_factor", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("resonance_frequency", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("load_distribution_parameter", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("defect_type", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("defect_initial_position", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("B_sig", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("step_size_sig", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("duration_sig", type=parser.modul_parameter, required=True, location="json")

        # 数据库参数
        rp.add_argument("file_lists", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("label_lists", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("length", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("shift", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("save_option", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("function_option", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("signal_option", type=parser.modul_parameter, required=True, location="json")

        # 2.3 模块二参数
        rp.add_argument("translation", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("rotation", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("noise", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("scale", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("image_multi", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("image_deltax", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("image_deltay", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("image_rot", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("image_snr", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("image_rescale", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("GAN_faulttype", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("GAN_rot_fre", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("GAN_num", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("GAN_numEpochs", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("GAN_Z_dim", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("GAN_n_ball", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("GAN_d_ball", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("GAN_D_pitch", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("GAN_alpha", type=parser.modul_parameter, required=True, location="json")

        rp.add_argument("mont_mode", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("mont_distribution", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("mont_function_select", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("mont_m", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("mont_a", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("mont_b", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("mont_c", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("mont_d", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("mont_e", type=parser.modul_parameter, required=True, location="json")

        # 2.4 模块三参数
        rp.add_argument("DWT_class_of_filter", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("DWT_sf_mode", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("DWT_sf_nums", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("DWT_kurt_k", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("DWT_kurt_mode", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("DWT_kurt_nums", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("DWT_fs_mode", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("DWT_fs_F", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("DWT_fs_max", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("DWT_fs_min", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("DWT_threshold_mode", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("DWT_threshold_nums", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("DWT_threshold_method", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("DWT_threshold_coeff", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("EMD_max_length_IMF", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("EMD_min_length_peaks", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("EMD_sift_min_relative_tolerance", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("EMD_sift_max_iterations", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("EMD_max_energy_ratio", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("EMD_selected_levels", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("FFT_fs", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("FFT_critical_freqs_min", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("FFT_critical_freqs_max", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("FFT_mode", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("FFT_order", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("mean_nums", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("mean_filt_length", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("waveletpacket_class_of_filter", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("waveletpacket_sf_mode", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("waveletpacket_sf_nums", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("waveletpacket_kurt_mode", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("waveletpacket_kurt_nums", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("waveletpacket_kurt_k", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("waveletpacket_fs_mode", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("waveletpacket_fs_F", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("waveletpacket_fs_max", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("waveletpacket_fs_min", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("fast_kurtogram_nlevel", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("fast_kurtogram_fs", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("fast_kurtogram_mode", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("fast_kurtogram_order", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("fast_kurtogram_Kurtosis_figure", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("PSD_fr", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("PSD_n_ball", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("PSD_d_ball", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("PSD_D_pitch", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("PSD_alpha", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("PSD_frequency_band_max", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("PSD_factor", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("PSD_sideband", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("PSD_sampling_frequency", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("PSD_cut_off_frequency", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("PSD_filter_method", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("PSD_filter_order", type=parser.modul_parameter, required=True, location="json")

        rp.add_argument("baring_dynamics_simulation", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("no_defect_type", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("outer_ring_defect_type", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("inner_ring_defect_type", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("ball_defect_type", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("segmentaion_len", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("segmentaion_nums", type=parser.modul_parameter, required=True, location="json")

        # 2.5 模块四参数
        rp.add_argument("f_DWT_Energe_Entropy", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_DWT_Singular_Entropy", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_WaveletPacket_EnergyEntropy", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_WaveletPacket_Singular_Entropy", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_OPFCF", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_SBN", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_frequency_normal_features", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_EMD", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_FCF_ratio", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("feature_selection_method", type=parser.modul_parameter, required=True, location="json")

        # 普通频域特征选择
        rp.add_argument("f_features_max", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_features_min", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_features_mean", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_features_root_mean_square", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_features_standard_deviation", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_features_variance", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_features_median", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_features_skewness", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_features_kurtosis", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_features_peak_to_peak_value", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_features_crest_factor", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_features_shape_factor", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_features_impulse_factor", type=parser.modul_parameter, required=True, location="json")

        # 普通时与特征提取
        rp.add_argument("time_features_max", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("time_features_min", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("time_features_mean", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("time_features_root_mean_square", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("time_features_standard_deviation", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("time_features_variance", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("time_features_median", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("time_features_skewness", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("time_features_kurtosis", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("time_features_peak_to_peak_value", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("time_features_crest_factor", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("time_features_shape_factor", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("time_features_impulse_factor", type=parser.modul_parameter, required=True, location="json")

        # 各频域特征详细参数
        rp.add_argument("t_features", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("f_features", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("emd_fr", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("emd_n_ball", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("emd_d_ball", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("emd_d_pitch", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("emd_alpha", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("emd_fs", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("emd_fault_type", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("emd_n", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("emd_limit", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("emd_order", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("FCF_ratio_nlevel", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("FCF_ratio_order", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("FCF_ratio_fs", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("FCF_ratio_fr", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("FCF_ratio_n_ball", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("FCF_ratio_d_ball", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("FCF_ratio_pitch", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("FCF_ratio_alpha", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("FCF_ratio_image", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("DWT_EnergyEntropy_mode", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("DWT_EnergyEntropy_nums", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("DWT_SingularEntropy_mode", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("DWT_SingularEntropy_nums", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("WaveletPacket_EnergyEntropy_mode", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("WaveletPacket_EnergyEntropy_nums", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("WaveletPacket_SingularEntropy_mode", type=parser.modul_parameter, required=True,
                        location="json")
        rp.add_argument("WaveletPacket_SingularEntropy_nums", type=parser.modul_parameter, required=True,
                        location="json")
        rp.add_argument("OPFCF_BPFO", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("OPFCF_BPFI", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("OPFCF_BSF", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("OPFCF_FTF", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("OPFCF_fr", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("OPFCF_order", type=parser.modul_parameter, required=True, location="json")
        # rp.add_argument("OPFCF_num", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("OPFCF_fs", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("OPFCF_switch", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("OPFCF_delta_f0", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("OPFCF_threshold", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("OPFCF_k", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("OPFCF_n_ball", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("OPFCF_d_ball", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("OPFCF_d_pitch", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("OPFCF_alpha", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("OPFCF_RUL_image", type=parser.modul_parameter, required=True, location="json")

        # 特征选择参数
        rp.add_argument("feature_selection", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("svd_switch", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("svd_dimension", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("pca_switch", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("pca_method", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("pca_dimension_method", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("pca_dimension", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("pca_percent", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("fda_switch", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("fda_dimension", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("AE_switch", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("AE_dimension", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Monotonicity_switch", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Correlation_switch", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Monotonicity_threshold", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Correlation_threshold", type=parser.modul_parameter, required=True, location="json")

        # 2.6 模块五参数
        rp.add_argument("opt_algorithm", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("rul_pre", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("pso_pop_size", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("pso_max_itr", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("ga_pop_size", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("ga_max_itr", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("sa_alpha", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("sa_max_itr", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("c_svm", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("gamma_svm", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("k_knn", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("weights_knn", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("max_depth_dt", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("max_leaf_nodes_dt", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("max_depth_rf", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("max_leaf_nodes_rf", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("n_estimators_rf", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("Dropout_dbn", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("LearningRate_RBM_dbn", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("LearningRate_nn_dbn", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("LayerCount_ae", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("units1_ae", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("units2_ae", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("units3_ae", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("epochs_ae", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("batchSize_ae", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("denseActivation_ae", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("max_depth_et", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("max_leaf_nodes_et", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("n_estimators_et", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("n_estimators_bagging", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("max_leaf_nodes_bagging", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("dropout_cnn", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("learning_rate_cnn", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("batch_size_cnn", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("conv_cnn", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("LSTMCount_lstm", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("units1_lstm", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("units2_lstm", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("units3_lstm", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("dropoutRate_lstm", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("epochs_lstm", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("batchSize_lstm", type=parser.modul_parameter, required=True, location="json")
        rp.add_argument("denseActivation_lstm", type=parser.modul_parameter, required=True, location="json")

        # 2.7 其他参数

        # 2.8 参数校验
        ret = rp.parse_args()
        modul1 = ret["bearing_simulation"]
        modul2 = ret["data_augmentation"]
        modul3 = ret["data_preprocessing"]
        modul4 = ret["features_extraction"]
        modul5 = ret["fault_diagnose"]

        output_files = ret["output_files"]
        output_images = ret["output_images"]

        # 物理模型参数
        Do_phy = ret["Do_phy"]
        Di_phy = ret["Di_phy"]
        d_phy = ret["d_phy"]
        Kb_phy = ret["Kb_phy"]
        alpha_phy = ret["alpha_phy"]
        Nb = ret["Nb"]
        Ms = ret["Ms"]
        Mp = ret["Mp"]
        Mr = ret["Mr"]
        Ks = ret["Ks"]
        Kp = ret["Kp"]
        Kr = ret["Kr"]
        Cs = ret["Cs"]
        Cp = ret["Cp"]
        Cr = ret["Cr"]
        L = ret["L"]
        B_phy = ret["B_phy"]
        H = ret["H"]
        ORS = ret["ORS"]
        ORN = ret["ORN"]
        OR_defect_position = ret["OR_defect_position"]
        IRS = ret["IRS"]
        IRN = ret["IRN"]
        IR_defect_position = ret["IR_defect_position"]
        BS = ret["BS"]
        BN = ret["BN"]
        B_defect_position = ret["B_defect_position"]
        B_defect_identifier = ret["B_defect_identifier"]
        Fr = ret["Fr"]
        Fa = ret["Fa"]
        Omega_shaft = ret["Omega_shaft"]
        step_size_phy = ret["step_size_phy"]
        duration_phy = ret["duration_phy"]
        mutation_percentage = ret["mutation_percentage"]
        initial_angular_position = ret["initial_angular_position"]

        # 信号参数
        Do_sig = ret["Do_sig"]
        Di_sig = ret["Di_sig"]
        z_sig = ret["z_sig"]
        D_sig = ret["D_sig"]
        type_factor_sig = ret["type_factor_sig"]
        alpha_sig = ret["alpha_sig"]
        load_max = ret["load_max"]
        shaft_speed = ret["shaft_speed"]
        phi_limit = ret["phi_limit"]
        load_proportional_factor = ret["load_proportional_factor"]
        resonance_frequency = ret["resonance_frequency"]
        load_distribution_parameter = ret["load_distribution_parameter"]
        defect_type = ret["defect_type"]
        defect_initial_position = ret["defect_initial_position"]
        B_sig = ret["B_sig"]
        step_size_sig = ret["step_size_sig"]
        duration_sig = ret["duration_sig"]

        # 数据库参数
        file_lists = ret["file_lists"]
        label_lists = ret["label_lists"]
        length = ret["length"]
        shift = ret["shift"]
        save_option = ret["save_option"]
        function_option = ret["function_option"]
        signal_option = ret["signal_option"]

        ###########modul2############
        translation = ret["translation"]
        rotation = ret["rotation"]
        noise = ret["noise"]
        scale = ret["scale"]
        image_multi = ret["image_multi"]
        image_deltax = ret["image_deltax"]
        image_deltay = ret["image_deltay"]
        image_rot = ret["image_rot"]
        image_snr = ret["image_snr"]
        image_rescale = ret["image_rescale"]
        GAN_faulttype = ret["GAN_faulttype"]
        GAN_rot_fre = ret["GAN_rot_fre"]
        GAN_num = ret["GAN_num"]
        GAN_numEpochs = ret["GAN_numEpochs"]
        GAN_Z_dim = ret["GAN_Z_dim"]
        GAN_n_ball = ret["GAN_n_ball"]
        GAN_d_ball = ret["GAN_d_ball"]
        GAN_D_pitch = ret["GAN_D_pitch"]
        GAN_alpha = ret["GAN_alpha"]
        mont_mode = ret["mont_mode"]
        mont_distribution = ret["mont_distribution"]
        mont_function_select = ret["mont_function_select"]
        mont_m = ret["mont_m"]
        mont_a = ret["mont_a"]
        mont_b = ret["mont_b"]
        mont_c = ret["mont_c"]
        mont_d = ret["mont_d"]
        mont_e = ret["mont_e"]

        ###########modul3############
        DWT_class_of_filter = ret["DWT_class_of_filter"]
        DWT_sf_mode = ret["DWT_sf_mode"]
        DWT_sf_nums = ret["DWT_sf_nums"]
        DWT_kurt_mode = ret["DWT_kurt_mode"]
        DWT_kurt_nums = ret["DWT_kurt_nums"]
        DWT_kurt_k = ret["DWT_kurt_k"]
        DWT_fs_mode = ret["DWT_fs_mode"]
        DWT_fs_F = ret["DWT_fs_F"]
        DWT_fs_max = ret["DWT_fs_max"]
        DWT_fs_min = ret["DWT_fs_min"]
        DWT_threshold_mode = ret["DWT_threshold_mode"]
        DWT_threshold_nums = ret["DWT_threshold_nums"]
        DWT_threshold_method = ret["DWT_threshold_method"]
        DWT_threshold_coeff = ret["DWT_threshold_coeff"]
        EMD_max_length_IMF = ret["EMD_max_length_IMF"]
        EMD_min_length_peaks = ret["EMD_min_length_peaks"]
        EMD_sift_min_relative_tolerance = ret["EMD_sift_min_relative_tolerance"]
        EMD_sift_max_iterations = ret["EMD_sift_max_iterations"]
        EMD_max_energy_ratio = ret["EMD_max_energy_ratio"]
        EMD_selected_levels = ret["EMD_selected_levels"]
        FFT_fs = ret["FFT_fs"]
        FFT_critical_freqs_min = ret["FFT_critical_freqs_min"]
        FFT_critical_freqs_max = ret["FFT_critical_freqs_max"]
        FFT_mode = ret["FFT_mode"]
        FFT_order = ret["FFT_order"]
        mean_nums = ret["mean_nums"]
        mean_filt_length = ret["mean_filt_length"]
        waveletpacket_class_of_filter = ret["waveletpacket_class_of_filter"]
        waveletpacket_sf_mode = ret["waveletpacket_sf_mode"]
        waveletpacket_sf_nums = ret["waveletpacket_sf_nums"]
        waveletpacket_kurt_mode = ret["waveletpacket_kurt_mode"]
        waveletpacket_kurt_nums = ret["waveletpacket_kurt_nums"]
        waveletpacket_kurt_k = ret["waveletpacket_kurt_k"]
        waveletpacket_fs_mode = ret["waveletpacket_fs_mode"]
        waveletpacket_fs_F = ret["waveletpacket_fs_F"]
        waveletpacket_fs_max = ret["waveletpacket_fs_max"]
        waveletpacket_fs_min = ret["waveletpacket_fs_min"]

        fast_kurtogram_nlevel = ret["fast_kurtogram_nlevel"]
        fast_kurtogram_fs = ret["fast_kurtogram_fs"]
        fast_kurtogram_mode = ret["fast_kurtogram_mode"]
        fast_kurtogram_order = ret["fast_kurtogram_order"]
        fast_kurtogram_Kurtosis_figure = ret["fast_kurtogram_Kurtosis_figure"]
        PSD_fr = ret["PSD_fr"]
        PSD_n_ball = ret["PSD_n_ball"]
        PSD_d_ball = ret["PSD_d_ball"]
        PSD_D_pitch = ret["PSD_D_pitch"]
        PSD_alpha = ret["PSD_alpha"]
        PSD_frequency_band_max = ret["PSD_frequency_band_max"]
        PSD_factor = ret["PSD_factor"]
        PSD_sideband = ret["PSD_sideband"]
        PSD_sampling_frequency = ret["PSD_sampling_frequency"]
        PSD_cut_off_frequency = ret["PSD_cut_off_frequency"]
        PSD_filter_method = ret["PSD_filter_method"]
        PSD_filter_order = ret["PSD_filter_order"]

        # 数据切割

        baring_dynamics_simulation = ret["baring_dynamics_simulation"]
        no_defect_type = ret["no_defect_type"]
        outer_ring_defect_type = ret["outer_ring_defect_type"]
        inner_ring_defect_type = ret["inner_ring_defect_type"]
        ball_defect_type = ret["ball_defect_type"]
        segmentaion_len = ret["segmentaion_len"]
        segmentaion_nums = ret["segmentaion_nums"]

        ##########modul4############
        f_DWT_Energe_Entropy = ret["f_DWT_Energe_Entropy"]
        f_DWT_Singular_Entropy = ret["f_DWT_Singular_Entropy"]
        f_WaveletPacket_EnergyEntropy = ret["f_WaveletPacket_EnergyEntropy"]
        f_WaveletPacket_Singular_Entropy = ret["f_WaveletPacket_Singular_Entropy"]
        f_OPFCF = ret["f_OPFCF"]
        f_SBN = ret["f_SBN"]
        f_frequency_normal_features = ret["f_frequency_normal_features"]
        f_EMD = ret["f_EMD"]
        f_FCF_ratio = ret["f_FCF_ratio"]
        feature_selection_method = ret["feature_selection_method"]

        # 普通频域特征选择
        f_features_max = ret["f_features_max"]
        f_features_min = ret["f_features_min"]
        f_features_mean = ret["f_features_mean"]
        f_features_root_mean_square = ret["f_features_root_mean_square"]
        f_features_standard_deviation = ret["f_features_standard_deviation"]
        f_features_variance = ret["f_features_variance"]
        f_features_median = ret["f_features_median"]
        f_features_skewness = ret["f_features_skewness"]
        f_features_kurtosis = ret["f_features_kurtosis"]
        f_features_peak_to_peak_value = ret["f_features_peak_to_peak_value"]
        f_features_crest_factor = ret["f_features_crest_factor"]
        f_features_shape_factor = ret["f_features_shape_factor"]
        f_features_impulse_factor = ret["f_features_impulse_factor"]

        # 普通时域特征选择
        time_features_max = ret["time_features_max"]
        time_features_min = ret["time_features_min"]
        time_features_mean = ret["time_features_mean"]
        time_features_root_mean_square = ret["time_features_root_mean_square"]
        time_features_standard_deviation = ret["time_features_standard_deviation"]
        time_features_variance = ret["time_features_variance"]
        time_features_median = ret["time_features_median"]
        time_features_skewness = ret["time_features_skewness"]
        time_features_kurtosis = ret["time_features_kurtosis"]
        time_features_peak_to_peak_value = ret["time_features_peak_to_peak_value"]
        time_features_crest_factor = ret["time_features_crest_factor"]
        time_features_shape_factor = ret["time_features_shape_factor"]
        time_features_impulse_factor = ret["time_features_impulse_factor"]

        # 各频域特征详细参数
        t_features = ret["t_features"]
        f_features = ret["f_features"]
        emd_fr = ret["emd_fr"]
        emd_n_ball = ret["emd_n_ball"]
        emd_d_ball = ret["emd_d_ball"]
        emd_d_pitch = ret["emd_d_pitch"]
        emd_alpha = ret["emd_alpha"]
        emd_fs = ret["emd_fs"]
        emd_fault_type = ret["emd_fault_type"]
        emd_n = ret["emd_n"]
        emd_limit = ret["emd_limit"]
        emd_order = ret["emd_order"]
        FCF_ratio_nlevel = ret["FCF_ratio_nlevel"]
        FCF_ratio_order = ret["FCF_ratio_order"]
        FCF_ratio_fs = ret["FCF_ratio_fs"]
        FCF_ratio_fr = ret["FCF_ratio_fr"]
        FCF_ratio_n_ball = ret["FCF_ratio_n_ball"]
        FCF_ratio_d_ball = ret["FCF_ratio_d_ball"]
        FCF_ratio_pitch = ret["FCF_ratio_pitch"]
        FCF_ratio_alpha = ret["FCF_ratio_alpha"]
        FCF_ratio_image = ret["FCF_ratio_image"]
        DWT_EnergyEntropy_mode = ret["DWT_EnergyEntropy_mode"]
        DWT_EnergyEntropy_nums = ret["DWT_EnergyEntropy_nums"]
        DWT_SingularEntropy_mode = ret["DWT_SingularEntropy_mode"]
        DWT_SingularEntropy_nums = ret["DWT_SingularEntropy_nums"]
        WaveletPacket_EnergyEntropy_mode = ret["WaveletPacket_EnergyEntropy_mode"]
        WaveletPacket_EnergyEntropy_nums = ret["WaveletPacket_EnergyEntropy_nums"]
        WaveletPacket_SingularEntropy_mode = ret["WaveletPacket_SingularEntropy_mode"]
        WaveletPacket_SingularEntropy_nums = ret["WaveletPacket_SingularEntropy_nums"]
        OPFCF_BPFO = ret["OPFCF_BPFO"]
        OPFCF_BPFI = ret["OPFCF_BPFI"]
        OPFCF_BSF = ret["OPFCF_BSF"]
        OPFCF_FTF = ret["OPFCF_FTF"]
        OPFCF_fr = ret["OPFCF_fr"]
        OPFCF_order = ret["OPFCF_order"]
        # OPFCF_num = ret["OPFCF_num"]
        OPFCF_fs = ret["OPFCF_fs"]
        OPFCF_switch = ret["OPFCF_switch"]
        OPFCF_delta_f0 = ret["OPFCF_delta_f0"]
        OPFCF_threshold = ret["OPFCF_threshold"]
        OPFCF_k = ret["OPFCF_k"]
        OPFCF_n_ball = ret["OPFCF_n_ball"]
        OPFCF_d_ball = ret["OPFCF_d_ball"]
        OPFCF_d_pitch = ret["OPFCF_d_pitch"]
        OPFCF_alpha = ret["OPFCF_alpha"]
        OPFCF_RUL_image = ret["OPFCF_RUL_image"]

        # 特征选择参数
        feature_selection = ret["feature_selection"]
        svd_switch = ret["svd_switch"]
        svd_dimension = ret["svd_dimension"]
        pca_switch = ret["pca_switch"]
        pca_method = ret["pca_method"]
        pca_dimension_method = ret["pca_dimension_method"]
        pca_dimension = ret["pca_dimension"]
        pca_percent = ret["pca_percent"]
        fda_switch = ret["fda_switch"]
        fda_dimension = ret["fda_dimension"]
        AE_switch = ret["AE_switch"]
        AE_dimension = ret["AE_dimension"]
        Monotonicity_switch = ret["Monotonicity_switch"]
        Correlation_switch = ret["Correlation_switch"]
        Monotonicity_threshold = ret["Monotonicity_threshold"]
        Correlation_threshold = ret["Correlation_threshold"]

        ######## modul5 ##########
        opt_algorithm = ret["opt_algorithm"]
        rul_pre = ret["rul_pre"]
        pso_pop_size = ret["pso_pop_size"]
        pso_max_itr = ret["pso_max_itr"]
        ga_pop_size = ret["ga_pop_size"]
        ga_max_itr = ret["ga_max_itr"]
        sa_alpha = ret["sa_alpha"]
        sa_max_itr = ret["sa_max_itr"]
        c_svm = ret["c_svm"]
        gamma_svm = ret["gamma_svm"]
        k_knn = ret["k_knn"]
        weights_knn = ret["weights_knn"]
        max_depth_dt = ret["max_depth_dt"]
        max_leaf_nodes_dt = ret["max_leaf_nodes_dt"]
        max_depth_rf = ret["max_depth_rf"]
        max_leaf_nodes_rf = ret["max_leaf_nodes_rf"]
        n_estimators_rf = ret["n_estimators_rf"]
        Dropout_dbn = ret["Dropout_dbn"]
        LearningRate_RBM_dbn = ret["LearningRate_RBM_dbn"]
        LearningRate_nn_dbn = ret["LearningRate_nn_dbn"]
        LayerCount_ae = ret["LayerCount_ae"]
        units1_ae = ret["units1_ae"]
        units2_ae = ret["units2_ae"]
        units3_ae = ret["units3_ae"]
        epochs_ae = ret["epochs_ae"]
        batchSize_ae = ret["batchSize_ae"]
        denseActivation_ae = ret["denseActivation_ae"]
        max_depth_et = ret["max_depth_et"]
        max_leaf_nodes_et = ret["max_leaf_nodes_et"]
        n_estimators_et = ret["n_estimators_et"]
        n_estimators_bagging = ret["n_estimators_bagging"]
        max_leaf_nodes_bagging = ret["max_leaf_nodes_bagging"]
        dropout_cnn = ret["dropout_cnn"]
        learning_rate_cnn = ret["learning_rate_cnn"]
        batch_size_cnn = ret["batch_size_cnn"]
        conv_cnn = ret["conv_cnn"]
        LSTMCount_lstm = ret["LSTMCount_lstm"]
        units1_lstm = ret["units1_lstm"]
        units2_lstm = ret["units2_lstm"]
        units3_lstm = ret["units3_lstm"]
        dropoutRate_lstm = ret["dropoutRate_lstm"]
        epochs_lstm = ret["epochs_lstm"]
        batchSize_lstm = ret["batchSize_lstm"]
        denseActivation_lstm = ret["denseActivation_lstm"]

        # 2.8 user_id通过g对象获取
        user_id = g.user_id

        # 判断流程选项是否正确:1, 2, 3, 4, 5, 12, 23, 123, 34, 45, 345
        process_validation(modul1, modul2, modul3, modul4, modul5)

        # 2. 清空用户文档
        try:
            shutil.rmtree(f"/home/python/Desktop/PHM_Dev/user_file/{user_id}")
            os.makedirs(f"/home/python/Desktop/PHM_Dev/user_file/{user_id}")

        except:
            os.makedirs(f"/home/python/Desktop/PHM_Dev/user_file/{user_id}")

        try:
            os.makedirs(f"/home/python/Desktop/PHM_Dev/upload_file/{user_id}")
        except:
            pass
        try:
            os.makedirs(f"/home/python/Desktop/PHM_Dev/upload_file/{user_id}/upload")
        except:
            pass

        # 3.逻辑处理
        # TODO:添加流程模块
        print("模块选择为:", modul1, modul2, modul3, modul4, modul5)
        time_data = 0

        save_path = f"/home/python/Desktop/PHM_Dev/user_file/{user_id}"
        upload_file = f"/home/python/Desktop/PHM_Dev/upload_file/{user_id}"

        # 特殊流程:蒙特卡洛部分
        if modul2 == 3:
            print("mont_mode", mont_mode)
            if mont_mode == 0:
                mont_file_path = "/home/python/Desktop/PHM_Dev/moduls/data_augmentation/folder"
            else:
                mont_file_path = "/home/python/Desktop/PHM_Dev/moduls/data_augmentation/theta.mat"
            print("mont_file_path", mont_file_path)
            monte_results = Monte_Carlo_DA(mont_file_path,
                                           save_path=save_path,
                                           mode=mont_mode,
                                           distribution=mont_distribution,
                                           function_select=mont_function_select,
                                           m=mont_m, a=mont_a, b=mont_b,
                                           c=mont_c, d=mont_d, e=mont_e,
                                           output_file=output_files, output_image=output_images)

            # 5 将所有保存的文件打包
            from utils.zip_dir import zip_dir

            out_fullname = f"/home/python/Desktop/PHM_Dev/user_file/{user_id}.zip"
            dirpath = f"/home/python/Desktop/PHM_Dev/user_file/{user_id}"
            zip_dir(dirpath, out_fullname)

            # 3.1 对方法列表进行拆包
            # 3.2 将对方的参数变量全部存入指定数据库
            # 3.3 根据制定法功法调取制定函数方法
            # 4.返回值:更新用户数据库下载状态

            # UserParameter.query.filter_by(user_id=user_id).update({'download_status': True})

            try:
                db.session.commit()
            except Exception as ex:
                db.session.rollback()
                return {"message": ex}, 507
            #
            # download_status = UserParameter.query.options(load_only(UserParameter.download_status)).filter(
            #     UserParameter.user_id == user_id).first()
            # # print(download_status)
            return {"message": "completion of task!",
                    "data": {
                        "download_status": True
                    }}


        ##############################################################
        # 新流程框架
        else:
            # 1:读取数据:如无数据可读,则use_data,use_data_label=None.
            # 1.1 读取信号
            from moduls.utils.read_data import read_time_domain_signal
            try:
                file_name = 'time_domain_signal.mat'
                data_path = os.path.join(upload_file, "upload")
                data_path = os.path.join(data_path, file_name)
                use_data = read_time_domain_signal(data_path)
            except:
                use_data = None

            # 1.2 读取label:读取数据label:允许不上传label,并且label与data必须匹配
            if use_data:
                try:
                    label_file_name = 'data_label.mat'
                    label_path = os.path.join(upload_file, "upload")
                    label_path = os.path.join(label_path, label_file_name)
                    use_data_label = read_time_domain_signal(label_path)
                    print("读取label为:", use_data_label)

                except:
                    use_data_label = None

                else:
                    # 判断数据与label形状是否匹配
                    if use_data.shape[0] != use_data_label.shape[0]:
                        abort(400, "The data and label are not suitable. Please check the data file again.")
            else:
                use_data = None
                use_data_label = None

            # 2: 模块一
            # 用户自己上传数据情况
            if modul1 == 1:
                abort(400, "功能尚在维护.")
                from moduls.bearing_simulation.bearing_simulation_with_defects.bearing_simulation_with_defects import \
                    bearing_main
                from moduls.utils.utils import str_to_float, str_to_int
                B_defect_identifier = str_to_int(B_defect_identifier)
                OR_defect_position = str_to_float(OR_defect_position)
                IR_defect_position = str_to_float(IR_defect_position)
                B_defect_position = str_to_float(B_defect_position)

                use_data = bearing_main(Kb=Kb_phy,
                                        d=d_phy,
                                        Nb=Nb,
                                        Di=Di_phy,
                                        Do=Do_phy,
                                        contact_angle=alpha_phy,
                                        Mp=Mp,
                                        Kp=Kp,
                                        Cp=Cp,
                                        Ms=Ms,
                                        Ks=Ks,
                                        Cs=Cs,
                                        Mr=Mr,
                                        Kr=Kr,
                                        Cr=Cr,
                                        Fr=Fr,
                                        Fa=Fa,
                                        omega_shaft=Omega_shaft,
                                        L=L,
                                        B=B_phy,
                                        H=H,
                                        outer_ring_switch=ORS,
                                        outer_ring_number=ORN,
                                        outer_ring_local_position=OR_defect_position,
                                        inner_ring_switch=IRS,
                                        inner_ring_number=IRN,
                                        inner_ring_local_position=IR_defect_position,
                                        ball_switch=BS,
                                        ball_number=BN,
                                        ball_fault_ball_identifier=B_defect_identifier,
                                        ball_local_position=B_defect_position,
                                        mutation_percentage=mutation_percentage,
                                        initial_angular_position=initial_angular_position,
                                        sim_duration=duration_phy,
                                        step_size=step_size_phy,
                                        save_path=save_path,
                                        output_image=output_images,
                                        output_file=output_files
                                        )
                use_data = use_data.reshape((1, -1))
                print("物理模型输出形状为:", use_data.shape)
            elif modul1 == 2:
                from moduls.bearing_simulation.signal_based_bearing_defect_model.bearing_defect_simulation_main import \
                    signal_main
                defect_type = 0 if not defect_type else defect_type
                defect_type_list = []
                defect_type_list.append(defect_type)
                print(defect_type_list)
                use_data, use_data_label = signal_main(D=D_sig,
                                                       di=Di_sig,
                                                       do=Do_sig,
                                                       Z=z_sig,
                                                       contact_angle=alpha_sig,
                                                       bearing_type_factor=type_factor_sig,
                                                       load_max=load_max,
                                                       load_proportional_factor=load_proportional_factor,
                                                       shaft_speed=shaft_speed,
                                                       resonance_frequency=resonance_frequency,
                                                       phi_limit=phi_limit,
                                                       load_distribution_parameter=load_distribution_parameter,
                                                       defect_type_list=defect_type_list,
                                                       decaying_parameter=B_sig,
                                                       defect_initial_position=defect_initial_position,
                                                       step_size=step_size_sig,
                                                       duration=duration_sig,
                                                       save_path=save_path,
                                                       output_file=output_files,
                                                       output_image=output_images)
                print("信号模型输出形状为:", use_data.shape)

            elif modul1 == 11:
                from moduls.utils.utils import str_to_float, str_to_int
                print("label_lists", len(label_lists))
                file_lists = str_to_int(file_lists)
                if len(label_lists) == 0:
                    label_lists = []
                else:
                    label_lists = str_to_int(label_lists)
                print("label_lists", label_lists)
                from utils.database_read import CRWU_read

                use_data, use_data_label = CRWU_read(file_lists=file_lists,
                                                     file_path="/home/python/Desktop/Dataset/CRWU",
                                                     label_lists=label_lists,
                                                     length=length, shift=shift,
                                                     save_option=save_option,
                                                     output_file=output_files,
                                                     save_path=save_path)

            elif modul1 == 12:
                from moduls.utils.utils import str_to_float, str_to_int
                print("label_lists", len(label_lists))
                file_lists = str_to_int(file_lists)
                if len(label_lists) == 0:
                    label_lists = []
                else:
                    label_lists = str_to_int(label_lists)
                print("label_lists", label_lists)

                from utils.database_read import IMS_read
                use_data, use_data_label = IMS_read(file_lists=file_lists, file_path="/home/python/Desktop/Dataset/IMS",
                                                    label_lists=label_lists,
                                                    length=length, shift=shift,
                                                    function_option=function_option,
                                                    save_option=save_option,
                                                    output_file=output_files, save_path=save_path)
            elif modul1 == 13:
                from moduls.utils.utils import str_to_float, str_to_int
                file_lists = str_to_int(file_lists)
                from utils.database_read import FEMTO_read
                use_data, use_data_label = FEMTO_read(file_lists=file_lists,
                                                      file_path="/home/python/Desktop/Dataset/FEMTO",
                                                      length=length, shift=shift,
                                                      save_option=save_option,
                                                      output_file=output_files,
                                                      save_path=save_path)
            elif modul1 == 14:
                from moduls.utils.utils import str_to_float, str_to_int
                print("label_lists", len(label_lists))
                file_lists = str_to_int(file_lists)
                if len(label_lists) == 0:
                    label_lists = []
                else:
                    label_lists = str_to_int(label_lists)
                print("label_lists", label_lists)
                from utils.database_read import Paderborn_read
                use_data, use_data_label = Paderborn_read(file_lists=file_lists,
                                                          file_path="/home/python/Desktop/Dataset/Paderborn",
                                                          label_lists=label_lists,
                                                          length=length, shift=shift,
                                                          signal_option=signal_option,
                                                          function_option=function_option,
                                                          save_option=save_option,
                                                          output_file=output_files,
                                                          save_path=save_path)

            elif modul1 == 15:
                from moduls.utils.utils import str_to_float, str_to_int

                print("label_lists", len(label_lists))
                file_lists = str_to_int(file_lists)
                if len(label_lists) == 0:
                    label_lists = []
                else:
                    label_lists = str_to_int(label_lists)
                print("label_lists", label_lists)

                from utils.database_read import XJTU_read
                print("XJ:", file_lists)
                use_data, use_data_label = XJTU_read(file_lists=file_lists,
                                                     file_path="/home/python/Desktop/Dataset/XJTU",
                                                     label_lists=label_lists,
                                                     length=length, shift=shift,
                                                     function_option=function_option,
                                                     save_option=save_option,
                                                     output_file=output_files,
                                                     save_path=save_path)
            else:
                abort(400, "Modul 1 ERROR!")

            # 数据打乱
            index = [i for i in range(len(use_data))]
            np.random.shuffle(index)
            use_data = use_data[index]
            use_data_label = use_data_label[index]

            print("模块一输出数据形状为:", use_data.shape, use_data_label.shape)
            # 3 模块二
            # 模块二运行条件判断
            if modul2 and not use_data:
                abort(400, "No data found. Please check the upload file or procedure configuration.")
            if modul2:
                from moduls.data_augmentation.data_augmentation_main import data_augmentation_main
                switch = [translation, rotation, noise, scale]
                print(output_files, output_images)
                if modul2 == 1:
                    from moduls.utils.utils import str_to_float, str_to_int
                    image_deltax = str_to_int(image_deltax)
                    image_deltay = str_to_float(image_deltay)
                    image_rot = str_to_float(image_rot)
                    image_snr = str_to_float(image_snr)
                    image_rescale = str_to_float(image_rescale)
                    print(image_deltax, image_deltax, image_rot, image_snr, image_rescale)
                print(image_deltax, image_deltay, image_rot, image_snr, image_rescale)
                print("模块2输入数据的形状为:", use_data.shape)
                # 情况一:有label剧增
                if use_data_label:
                    use_data, use_data_label = data_augmentation_main(use_data,
                                                                      use_data_label,
                                                                      switch=modul2,
                                                                      faulttype=GAN_faulttype,
                                                                      rot_fre=GAN_rot_fre,
                                                                      GAN_num=GAN_num,
                                                                      numEpochs=GAN_numEpochs,
                                                                      Z_dim=GAN_Z_dim,
                                                                      n_ball=GAN_n_ball,
                                                                      d_ball=GAN_d_ball,
                                                                      D_pitch=GAN_D_pitch,
                                                                      alpha=GAN_alpha,
                                                                      multi=image_multi,
                                                                      deltax=image_deltax,
                                                                      deltay=image_deltay,
                                                                      rot=image_rot,
                                                                      snr=image_snr,
                                                                      rescale=image_rescale,
                                                                      image_transformation_DA_switch=switch,
                                                                      mode=0,
                                                                      distribution=0,
                                                                      m=100,
                                                                      p0=(1, 0),
                                                                      save_path=save_path,
                                                                      output_file=output_files,
                                                                      output_image=output_images)
                # 情况二:无label剧增
                else:
                    use_data, _ = data_augmentation_main(time_data,
                                                         None,
                                                         switch=modul2,
                                                         faulttype=GAN_faulttype,
                                                         rot_fre=GAN_rot_fre,
                                                         GAN_num=GAN_num,
                                                         numEpochs=GAN_numEpochs,
                                                         Z_dim=GAN_Z_dim,
                                                         n_ball=GAN_n_ball,
                                                         d_ball=GAN_d_ball,
                                                         D_pitch=GAN_D_pitch,
                                                         alpha=GAN_alpha,
                                                         multi=image_multi,
                                                         deltax=image_deltax,
                                                         deltay=image_deltay,
                                                         rot=image_rot,
                                                         snr=image_snr,
                                                         rescale=image_rescale,
                                                         image_transformation_DA_switch=switch,
                                                         mode=0,
                                                         distribution=0,
                                                         m=100,
                                                         p0=(1, 0),
                                                         save_path=save_path,
                                                         output_file=output_files,
                                                         output_image=output_images)

            # 4 模块三
            # 模块三运行条件判断:
            if modul3 and not use_data:
                abort(400, "No data found. Please check the upload file or procedure configuration.")

            if modul3:
                # if use_data.shape[1] <= 27:
                #     abort(400, "ValueError: The length of the input vector x must be greater than padlen, which is 27.")
                from moduls.data_denoise.data_denoise_main import data_denoise_main
                print("模块3输入数据形状为:", use_data.shape)
                use_data = data_denoise_main(use_data,
                                             switch=modul3,
                                             DWT_select=DWT_class_of_filter,
                                             DWT_Filter_name=DWT_sf_mode,
                                             DWT_Filter_N=DWT_sf_nums,
                                             DWT_Fs_name=DWT_fs_mode,
                                             DWT_Fs_F=DWT_fs_F,
                                             DWT_Fs_Fs=(DWT_fs_min, DWT_fs_max),
                                             DWT_kurtosis_max_name=DWT_kurt_mode,
                                             DWT_kurtosis_max_N=DWT_kurt_nums,
                                             DWT_kurtosis_max_k=DWT_kurt_k,
                                             DWT_threshold_name=DWT_threshold_mode,
                                             DWT_threshold_max_N=DWT_threshold_nums,
                                             threshold_method=DWT_threshold_method,
                                             threshold_coeff=DWT_threshold_coeff,
                                             EMD_max_length_IMF=EMD_max_length_IMF,
                                             EMD_min_length_peaks=EMD_min_length_peaks,
                                             EMD_sift_min_relative_tolerance=EMD_sift_min_relative_tolerance,
                                             EMD_sift_max_iterations=EMD_sift_max_iterations,
                                             EMD_max_energy_ratio=EMD_max_energy_ratio,
                                             EMD_selected_levels=EMD_selected_levels,
                                             FFT_fs=FFT_fs,
                                             FFT_critical_freqs=(FFT_critical_freqs_min, FFT_critical_freqs_max),
                                             FFT_mode=FFT_mode,
                                             FFT_order=FFT_order,
                                             Mean_n=mean_nums,
                                             Mean_filt_length=mean_filt_length,
                                             WaveletPacket_select=waveletpacket_class_of_filter,
                                             WaveletPacket_Filter_name=waveletpacket_sf_mode,
                                             WaveletPacket_Filter_N=waveletpacket_sf_nums,
                                             WaveletPacket_Fs_name=waveletpacket_fs_mode,
                                             WaveletPacket_Fs_sampling_frequency=waveletpacket_fs_F,
                                             WaveletPacket_Fs_band=(waveletpacket_fs_min, waveletpacket_fs_max),
                                             WaveletPacket_kurtosis_max_name=waveletpacket_kurt_mode,
                                             WaveletPacket_kurtosis_max_N=waveletpacket_kurt_nums,
                                             WaveletPacket_kurtosis_max_k=waveletpacket_kurt_k,
                                             PSD_filter_fr=PSD_fr,
                                             PSD_filter_n_ball=PSD_n_ball,
                                             PSD_filter_d_ball=PSD_d_ball,
                                             PSD_filter_d_pitch=PSD_D_pitch,
                                             PSD_filter_alpha=PSD_alpha,
                                             PSD_filter_frequency_band_max=PSD_frequency_band_max,
                                             PSD_filter_factor=PSD_factor,
                                             PSD_filter_sideband_switch=PSD_sideband,
                                             PSD_filter_sampling_frequency=PSD_sampling_frequency,
                                             PSD_filter_cut_off_frequency=PSD_cut_off_frequency,
                                             PSD_filter_filter_method=PSD_filter_method,
                                             PSD_filter_filter_order=PSD_filter_order,
                                             FK_nlevel=fast_kurtogram_nlevel,
                                             FK_Fs=fast_kurtogram_fs,
                                             FK_mode=fast_kurtogram_mode,  # "bandpass",
                                             FK_order=fast_kurtogram_order,
                                             Kurtosis_figure=fast_kurtogram_Kurtosis_figure,
                                             save_path=save_path,
                                             output_file=output_files,
                                             output_image=output_images)

            # 5 模块四
            # 模块四运行条件判断:
            if modul4 and (not use_data or not use_data_label):
                abort(400, "No data or label found. Please check the upload file or procedure configuration.")
            if modul4:

                print("模块4输入数据形状为:", use_data.shape, use_data_label.shape)
                from moduls.feature_extraction.All_features_main import All_feature_extraction

                f_features_list = [f_DWT_Energe_Entropy, f_DWT_Singular_Entropy, f_WaveletPacket_EnergyEntropy,
                                   f_WaveletPacket_Singular_Entropy, f_OPFCF, f_SBN, f_frequency_normal_features,
                                   f_EMD, f_FCF_ratio]
                print("f_features_list", f_features_list)

                f_normal_features = [f_features_max, f_features_min, f_features_mean, f_features_root_mean_square,
                                     f_features_standard_deviation, f_features_variance, f_features_median,
                                     f_features_skewness, f_features_kurtosis, f_features_peak_to_peak_value,
                                     f_features_crest_factor, f_features_shape_factor, f_features_impulse_factor]
                t_normal_features = [time_features_max, time_features_min, time_features_mean,
                                     time_features_root_mean_square,
                                     time_features_standard_deviation, time_features_variance, time_features_median,
                                     time_features_skewness, time_features_kurtosis, time_features_peak_to_peak_value,
                                     time_features_crest_factor, time_features_shape_factor,
                                     time_features_impulse_factor]

                print("t_normal_features", t_normal_features)

                use_data, use_data_label, name_list = All_feature_extraction(use_data,
                                                                             use_data_label,
                                                                             t_normal_features=t_normal_features,
                                                                             f_features_list=f_features_list,
                                                                             DWT_EnergyEntropy_name=DWT_EnergyEntropy_mode,
                                                                             DWT_EnergyEntropy_N=DWT_EnergyEntropy_nums,
                                                                             DWT_SingularEntropy_name=DWT_SingularEntropy_mode,
                                                                             DWT_SingularEntropy_N=DWT_SingularEntropy_nums,
                                                                             WaveletPacket_EnergyEntropy_name=WaveletPacket_EnergyEntropy_mode,
                                                                             WaveletPacket_EnergyEntropy_N=WaveletPacket_EnergyEntropy_nums,
                                                                             WaveletPacket_SingularEntropy_name=WaveletPacket_SingularEntropy_mode,
                                                                             WaveletPacket_SingularEntropy_N=WaveletPacket_SingularEntropy_nums,
                                                                             OPFCF_fault_type_list=(
                                                                                 OPFCF_BPFO, OPFCF_BPFI, OPFCF_BSF,
                                                                                 OPFCF_FTF),
                                                                             OPFCF_fr=OPFCF_fr,
                                                                             OPFCF_order=OPFCF_order,
                                                                             # OPFCF_num=OPFCF_num,
                                                                             OPFCF_fs=OPFCF_fs,
                                                                             OPFCF_switch=OPFCF_switch,
                                                                             OPFCF_delta_f0=OPFCF_delta_f0,
                                                                             OPFCF_threshold=OPFCF_threshold,
                                                                             OPFCF_k=OPFCF_k,
                                                                             OPFCF_n_ball=OPFCF_n_ball,
                                                                             OPFCF_d_ball=OPFCF_d_ball,
                                                                             OPFCF_d_pitch=OPFCF_d_pitch,
                                                                             OPFCF_alpha=OPFCF_alpha,
                                                                             OPFCF_RUL_image=OPFCF_RUL_image,
                                                                             f_normal_features=f_normal_features,
                                                                             emd_fr=emd_fr,
                                                                             emd_n_ball=emd_n_ball,
                                                                             emd_d_ball=emd_d_ball,
                                                                             emd_d_pitch=emd_d_pitch,
                                                                             emd_alpha=emd_alpha,
                                                                             emd_fs=emd_fs,
                                                                             emd_fault_type=emd_fault_type,
                                                                             emd_n=emd_n,
                                                                             emd_ord=emd_order,
                                                                             emd_limit=emd_limit,
                                                                             FCF_ratio_nlevel=FCF_ratio_nlevel,
                                                                             FCF_ratio_order=FCF_ratio_order,
                                                                             FCF_ratio_fs=FCF_ratio_fs,
                                                                             FCF_ratio_fr=FCF_ratio_fr,
                                                                             FCF_ratio_n_ball=FCF_ratio_n_ball,
                                                                             FCF_ratio_d_ball=FCF_ratio_d_ball,
                                                                             FCF_ratio_d_pitch=FCF_ratio_pitch,
                                                                             FCF_ratio_alpha=FCF_ratio_alpha,
                                                                             FCF_ratio_image=FCF_ratio_image,
                                                                             save_path=save_path,
                                                                             output_file=output_files
                                                                             )
                print("特征提取的输出结果形状为:", use_data.shape, use_data_label.shape)
                print("name_list:", name_list, type(name_list))
                use_data = np.nan_to_num(use_data)

                # 特征选择
                if feature_selection_method:
                    from moduls.feature_selection.Features_Selection_main import Features_Selection_main
                    use_data = Features_Selection_main(use_data,
                                                       use_data_label,
                                                       name_list,
                                                       Features_selection=feature_selection,
                                                       svd_switch=svd_switch,
                                                       svd_dimension=svd_dimension,
                                                       pca_switch=pca_switch,
                                                       pca_method=pca_method,
                                                       pca_dimension_method=pca_dimension_method,
                                                       pca_dimension=pca_dimension,
                                                       pca_percent=pca_percent,
                                                       fda_switch=fda_switch,
                                                       fda_dim=fda_dimension,
                                                       AE_switch=AE_switch,
                                                       AE_encoding_dim=AE_dimension,
                                                       Monotonicity_threshold=Monotonicity_threshold,
                                                       Monotonicity_switch=Monotonicity_switch,
                                                       Correlation_threshold=Correlation_threshold,
                                                       Correlation_switch=Correlation_switch,
                                                       save_path=save_path,
                                                       output_file=output_files,
                                                       output_image=output_images
                                                       )

                print("特征选择后use_data形状为:", use_data.shape)
            use_data = np.real(use_data)

            # 6 模块五
            # 模块五数据读取(新):----5情况下的 数据准备 与 条件判断
            x_train, y_train, x_test, y_test = None, None, None, None
            if modul5 and not (modul1 + modul2 + modul3 + modul4):
                from moduls.utils.read_data import read_time_domain_signal
                print("modul5: reading data!")
                traindata_name = 'traindata.mat'
                trainlabel_file_name = 'trainlabel.mat'
                testdata_name = 'testdata.mat'
                testlabel_file_name = 'testlabel.mat'
                traindata_path = os.path.join(upload_file, "upload")
                traindata_path = os.path.join(traindata_path, traindata_name)
                trainlabel_path = os.path.join(upload_file, "upload")
                trainlabel_path = os.path.join(trainlabel_path, trainlabel_file_name)
                testdata_path = os.path.join(upload_file, "upload")
                testdata_path = os.path.join(testdata_path, testdata_name)
                testlabel_path = os.path.join(upload_file, "upload")
                testlabel_path = os.path.join(testlabel_path, testlabel_file_name)
                x_train = read_time_domain_signal(traindata_path)
                y_train = read_time_domain_signal(trainlabel_path)
                x_test = read_time_domain_signal(testdata_path)
                y_test = read_time_domain_signal(testlabel_path)
                print("y_test", y_test.shape)
                # 报错:判断上传数据是否匹配
                if x_train.shape[0] != y_train.shape[0] or x_test.shape[0] != y_test.shape[0]:
                    abort(400, "The data and label are not suitable. Please check the data file again.")

            # 模块五:xxxx5情况下的 数据准备 与 条件判断
            elif modul5 and (not use_data or not use_data_label):
                abort(400, "No data or label found. Please check the upload file or procedure configuration.")

            # 使用流程中数据
            else:
                # 报错:故障诊断时,当样本数小于100报错.
                if not rul_pre:
                    if use_data.shape[0] < 100:
                        abort(400,
                              f"The number of training samples cannot be less than 100. The current sample size is {use_data.shape[0]}.")
                # 去NAN
                use_data = np.nan_to_num(use_data)
                use_data_label = np.nan_to_num(use_data_label)
                # 报错:故障诊断时,label种类必须>1
                if not rul_pre:
                    # print(list(set([1,])))
                    nums_of_labels = list(set(list(np.reshape(use_data_label, (-1,))))).__len__()
                    print("nums_of_labels", nums_of_labels)
                    if list(set(list(np.reshape(use_data_label, (-1,))))).__len__() < 2:
                        abort(400, f"The number of classes has to be greater than one; got {nums_of_labels} class.")

                try:
                    from sklearn.model_selection import train_test_split
                    x_train, x_test, y_train, y_test = train_test_split(use_data,
                                                                        use_data_label,
                                                                        test_size=0.25,
                                                                        random_state=0)
                except:
                    abort(400,
                          f"Data splitting error. Please check the data. The current data shape is {use_data.shape}. The current label shape is {use_data_label.shape}.")

            if modul5:
                """
                modul5:x_train特征: (nums, 200)
                modul5:y_train目标值: (nums, 1)"""


                print("x_train.shape", "y_train.shape", "x_test.shape", "y_test.shape")
                print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

                print("modul5:x_train特征:", x_train.shape, type(x_train))
                # print("modul5:x_train特征:", x_train[1, :])
                print("modul5:y_train目标值:", y_train.shape, type(y_train))
                # print("modul5:y_train目标值:", y_train[0, 0])

                save_data_func(x_train, output_file=output_files, save_path=save_path, file_name="traindata")
                save_data_func(y_train, output_file=output_files, save_path=save_path, file_name="trainlabel")
                save_data_func(x_test, output_file=output_files, save_path=save_path, file_name="testdata")
                save_data_func(y_test, output_file=output_files, save_path=save_path, file_name="testlabel")

                if opt_algorithm == 1:
                    option = "PSO"
                    if pso_pop_size < 1:
                        print("pso_pop_size", pso_pop_size)
                        abort(400, "The parameter PSO pop size must be a positive integer and greater than 2.")
                    if pso_max_itr < 1:
                        abort(400, "The parameter PSO maximum iteration must be a positive integer and greater than 2.")
                elif opt_algorithm == 2:
                    option = "GA"
                    print("ga_max_itr", ga_pop_size)
                    if ga_pop_size < 2:
                        abort(400, "The parameter GA pop size must be a positive integer and greater than 2.")
                    if ga_max_itr < 2:
                        abort(400, "The parameter GA maximum iteration must be a positive integer and greater than 2.")
                elif opt_algorithm == 3:
                    option = "SA"
                    if sa_max_itr < 2:
                        abort(400, "The parameter SA maximum iteration must be a positive integer and greater than 2.")
                    if sa_alpha <= 0 or sa_alpha >= 1:
                        abort(400, "The value range of parameter SA alpha is (0, 1).")
                else:
                    option = None

                if modul5 == 1:
                    from moduls.ml.knn_main import KNN
                    """
                    !pso_part_num=2, !pso_num_itr=5,
                    sa_initial_temp=500, sa_final_temp=1,
                    !sa_alpha=0.9, !sa_max_iter=5,
                    ga_threshold=100.9960, !ga_dna_size=9, !ga_pop_size=6, ga_cross_rate=0.3,
                    ga_mutation_rate=0.1, ga_n_generations=2,
                    参数范围：
                    k_range = range(1, 25)  # user input (under, upper bounds)
                    weight_choices = ["uniform", "distance"]  # user input, string in list
                    """

                    """
                    ga_pop_size = ret["ga_pop_size"]
                    ga_max_itr = ret["ga_max_itr"]
                    sa_alpha = ret["sa_alpha"]
                    sa_max_itr = ret["sa_max_itr"]"""
                    y_pre, params_dict = KNN(x_train, x_test, y_train, y_test,
                                             rul_pre=rul_pre,
                                             K=k_knn,
                                             weights=weights_knn,  # "distance",
                                             pso_part_num=pso_pop_size,
                                             pso_num_itr=pso_max_itr,
                                             sa_alpha=sa_alpha,
                                             sa_max_iter=sa_max_itr,
                                             ga_pop_size=ga_max_itr,
                                             ga_n_generations=ga_pop_size,
                                             opt_option=option,
                                             output_image=output_images,
                                             save_path=save_path)

                    # params_dict = {"K": K, "weights": weights}["accuracy"] = accuracy
                    report.word_knn(save_path=save_path,
                                    train_data=x_train, train_label=y_train, test_data=x_test, test_label=y_test,
                                    opt_algorithm=opt_algorithm, rul_pre=rul_pre,
                                    K=params_dict['K'], weights=params_dict['weights'],
                                    pso_pop_size=pso_pop_size, pso_max_itr=pso_max_itr,
                                    ga_pop_size=ga_pop_size, ga_max_itr=ga_max_itr,
                                    sa_alpha=sa_alpha, sa_max_itr=sa_max_itr,
                                    accuracy=params_dict['accuracy'],
                                    )
                elif modul5 == 2:
                    from moduls.ml.svm_main import SVM

                    # print(c_svm, gamma_svm)
                    y_pre, acc = SVM(x_train, x_test, y_train, y_test,
                                     rul_pre=rul_pre,
                                     C=c_svm,
                                     gamma=gamma_svm,
                                     pso_part_num=pso_pop_size,
                                     pso_num_itr=pso_max_itr,
                                     sa_alpha=sa_alpha,
                                     sa_max_iter=sa_max_itr,
                                     ga_pop_size=ga_max_itr,
                                     ga_n_generations=ga_pop_size,
                                     opt_option=option,
                                     output_image=output_images,
                                     save_path=save_path)
                    """params = [C, gamma]"""
                    # report.word_svm(c=c_svm, gamma=gamma_svm, save_path=save_path)
                elif modul5 == 3:
                    from moduls.ml.DT_main import DT

                    y_pre, acc = DT(x_train, x_test, y_train, y_test,
                                    rul_pre=rul_pre,
                                    max_depth=max_depth_dt,
                                    max_leaf_nodes=max_leaf_nodes_dt,
                                    sa_alpha=sa_alpha,
                                    sa_max_iter=sa_max_itr,
                                    ga_pop_size=ga_max_itr,
                                    ga_n_generations=ga_pop_size,
                                    opt_option=option,
                                    output_image=output_images,
                                    save_path=save_path)
                    """[max_depth, max_leaf_nodes]"""
                elif modul5 == 4:
                    from moduls.ml.RF_main import RF

                    y_pre, acc = RF(x_train, x_test, y_train, y_test,
                                    rul_pre=rul_pre,
                                    max_depth=max_depth_rf,
                                    max_leaf_nodes=max_leaf_nodes_rf,
                                    n_estimators=n_estimators_rf,
                                    sa_alpha=sa_alpha,
                                    sa_max_iter=sa_max_itr,
                                    ga_pop_size=ga_max_itr,
                                    ga_n_generations=ga_pop_size,
                                    opt_option=option,
                                    output_image=output_images,
                                    save_path=save_path)
                    """params = [max_depth, max_leaf_nodes, n_estimators]"""
                elif modul5 == 5:
                    # DBN数据处理
                    if rul_pre:
                        size_train = x_train.shape[0]
                        size_test = x_test.shape[0]
                        y_train = y_train.reshape(size_train, 1)
                        y_test = y_test.reshape(size_test, 1)
                    else:
                        y_train_max = int(max(y_train)) + 1
                        y_test_max = int(max(y_test)) + 1
                        from moduls.ml.dataset import to_cat
                        y_train = to_cat(y_train, num_classes=y_train_max)
                        y_test = to_cat(y_test, num_classes=y_test_max)

                    from moduls.ml.dbn_main import DBN
                    y_pre, acc = DBN(x_train, x_test, y_train, y_test,
                                     rul_pre=rul_pre,
                                     Dropout=Dropout_dbn,
                                     LearningRate_RBM=LearningRate_RBM_dbn,
                                     LearningRate_nn=LearningRate_nn_dbn,
                                     sa_alpha=sa_alpha,
                                     sa_max_iter=sa_max_itr,
                                     ga_pop_size=ga_max_itr,
                                     ga_n_generations=ga_pop_size,
                                     opt_option=option,
                                     output_image=output_images,
                                     save_path=save_path)
                    """params = [Dropout, LearningRate_RBM, LearningRate_nn]"""
                elif modul5 == 6:
                    # AE 数据处理
                    if rul_pre:
                        size_train = x_train.shape[0]
                        size_test = x_test.shape[0]
                        y_train = y_train.reshape(size_train, 1)
                        y_test = y_test.reshape(size_test, 1)
                    else:
                        y_train_max = int(max(y_train)) + 1
                        y_test_max = int(max(y_test)) + 1
                        from moduls.ml.dataset import to_cat
                        y_train = to_cat(y_train, num_classes=y_train_max)
                        y_test = to_cat(y_test, num_classes=y_test_max)

                    from moduls.ml.AE_main import AE
                    print("epochs_ae:", epochs_ae)
                    print("batchSize_ae:", batchSize_ae)
                    y_pre, acc = AE(x_train, x_test, y_train, y_test,
                                    rul_pre=rul_pre,
                                    layer_count=LayerCount_ae,
                                    units1=units1_ae,
                                    units2=units2_ae,
                                    units3=units3_ae,
                                    epochs=epochs_ae,
                                    batchSize=batchSize_ae,
                                    denseActivation=denseActivation_ae,
                                    optimizer='adam',
                                    sa_alpha=sa_alpha,
                                    sa_max_iter=sa_max_itr,
                                    ga_pop_size=ga_max_itr,
                                    ga_n_generations=ga_pop_size,
                                    opt_option=option,
                                    output_image=output_images,
                                    save_path=save_path)
                    """[layer_count, units1, units2, units3, epochs, batchSize, denseActivation, optimizer, "mae"]"""

                elif modul5 == 7:
                    from moduls.ml.ExtraTree_main import ET
                    y_pre, acc = ET(x_train, x_test, y_train, y_test,
                                    rul_pre=rul_pre,
                                    max_depth=max_depth_et,
                                    max_leaf_nodes=max_leaf_nodes_et,
                                    n_estimators=n_estimators_et,
                                    sa_alpha=sa_alpha,
                                    sa_max_iter=sa_max_itr,
                                    ga_pop_size=ga_max_itr,
                                    ga_n_generations=ga_pop_size,
                                    opt_option=option,
                                    output_image=output_images,
                                    save_path=save_path)
                    """params = [max_depth, max_leaf_nodes, n_estimators]"""
                elif modul5 == 8:
                    from moduls.ml.Bagging_main import Bagging
                    y_pre, acc = Bagging(x_train, x_test, y_train, y_test,
                                         rul_pre=rul_pre,
                                         max_leaf_nodes=max_leaf_nodes_bagging,
                                         n_estimators=n_estimators_bagging,
                                         sa_alpha=sa_alpha,
                                         sa_max_iter=sa_max_itr,
                                         ga_pop_size=ga_max_itr,
                                         ga_n_generations=ga_pop_size,
                                         opt_option=option,
                                         output_image=output_images,
                                         save_path=save_path)
                    """params = [max_depth, max_leaf_nodes, n_estimators]"""

                elif modul5 == 11:
                    from moduls.ml.cnn_main import CNN
                    # print("x_train", x_train.shape)
                    # print("x_train", x_train)
                    # print("y_train", y_train.shape)
                    # print("y_train", y_train)
                    """
                    参数范围：
                    dropout_cnn = ret["0.5"]
                    learning_rate_cnn = ret["0.002"]
                    batch_size_cnn = ret["128"]
                    conv_cnn = ret["6"]
                    """
                    # CNN数据处理
                    x_train = np.array(x_train)
                    x_test = np.array(x_test)
                    if rul_pre:
                        size_train = x_train.shape[0]
                        size_test = x_test.shape[0]
                        x_train = x_train.reshape(size_train, x_train.shape[1], 1)
                        x_test = x_test.reshape(size_test, x_test.shape[1], 1)
                        y_train = y_train.reshape(size_train, 1)
                        y_test = y_test.reshape(size_test, 1)
                    else:
                        size_train = x_train.shape[0]
                        size_test = x_test.shape[0]
                        x_train = x_train.reshape(size_train, x_train.shape[1], 1)
                        x_test = x_test.reshape(size_test, x_test.shape[1], 1)
                        print("后x_train", x_train.shape)
                        y_train_max = int(max(y_train)) + 1
                        y_test_max = int(max(y_test)) + 1
                        from moduls.ml.dataset import to_cat
                        y_train = to_cat(y_train, num_classes=y_train_max)
                        y_test = to_cat(y_test, num_classes=y_test_max)

                    # print("modul5:y_train目标值:", y_train)
                    if x_train.shape[1] < 12:
                        abort(400, "The minimum feature length of CNN is 12.")
                    if not option and (conv_cnn > 9 or conv_cnn < 1):
                        abort(400, "The range of the convolution kernel is [1, 9].")
                    if not option and (batch_size_cnn not in [1, 16, 32, 64, 128, 256]):
                        abort(400, "The size of Batch_size can only be in the range of [1, 16, 32, 64, 128, 256].")
                    y_pre, acc = CNN(x_train, x_test, y_train, y_test,
                                     rul_pre=rul_pre,
                                     dropout=dropout_cnn,
                                     learning_rate=learning_rate_cnn,
                                     batch_size=batch_size_cnn,
                                     conv=conv_cnn,
                                     sa_alpha=sa_alpha,
                                     sa_max_iter=sa_max_itr,
                                     ga_pop_size=ga_max_itr,
                                     ga_n_generations=ga_pop_size,
                                     opt_option=option,
                                     output_image=output_images,
                                     save_path=save_path)
                    """params = [dropout, learning_rate, batch_size, conv]"""
                elif modul5 == 12:
                    # LSTM的数据处理
                    if rul_pre:
                        size_train = x_train.shape[0]
                        size_test = x_test.shape[0]
                        x_train = x_train.reshape(size_train, x_train.shape[1], 1)
                        x_test = x_test.reshape(size_test, x_test.shape[1], 1)
                        y_train = y_train.reshape(size_train, 1)
                        y_test = y_test.reshape(size_test, 1)

                    else:
                        print("前x_train:", x_train.shape, type(x_train))
                        print("前y_train:", y_train.shape, type(y_train))
                        size_train = x_train.shape[0]
                        size_test = x_test.shape[0]
                        x_train = x_train.reshape(size_train, 1, x_train.shape[1])
                        x_test = x_test.reshape(size_test, 1, x_test.shape[1])
                        y_train_max = int(max(y_train)) + 1
                        y_test_max = int(max(y_test)) + 1
                        from moduls.ml.dataset import to_cat
                        y_train = to_cat(y_train, num_classes=y_train_max)
                        y_test = to_cat(y_test, num_classes=y_test_max)

                    print("x_train:", x_train.shape, x_train)
                    print("y_train:", y_train.shape, y_train)
                    from moduls.ml.lstm_main import LSTM
                    y_pre, acc = LSTM(x_train, x_test, y_train, y_test,
                                      rul_pre=rul_pre,
                                      layer_count=LSTMCount_lstm,
                                      units1=units1_lstm,
                                      units2=units2_lstm,
                                      units3=units3_lstm,
                                      dropoutRate=dropoutRate_lstm,
                                      epochs=epochs_lstm,
                                      batchSize=batchSize_lstm,
                                      denseActivation=denseActivation_lstm,
                                      sa_alpha=sa_alpha,
                                      sa_max_iter=sa_max_itr,
                                      ga_pop_size=ga_max_itr,
                                      ga_n_generations=ga_pop_size,
                                      opt_option=option,
                                      output_image=output_images,
                                      save_path=save_path)

                print("y_pre", y_pre.shape)
                save_data_func(y_pre, output_file=output_files, save_path=save_path, file_name="pred_label")
        # 5 将所有保存的文件打包
        from utils.zip_dir import zip_dir

        out_fullname = f"/home/python/Desktop/PHM_Dev/user_file/{user_id}.zip"
        dirpath = f"/home/python/Desktop/PHM_Dev/user_file/{user_id}"
        zip_dir(dirpath, out_fullname)

        # 3.1 对方法列表进行拆包
        # 3.2 将对方的参数变量全部存入指定数据库
        # 3.3 根据制定法功法调取制定函数方法
        # 4.返回值:更新用户数据库下载状态

        # UserParameter.query.filter_by(user_id=user_id).update({'download_status': True})
        print("快结束了!")
        try:
            db.session.commit()
        except Exception as ex:
            db.session.rollback()
            return {"message": ex}, 507

        # download_status = UserParameter.query.options(load_only(UserParameter.download_status)).filter(
        #     UserParameter.user_id == user_id).first()
        # print(download_status)
        return {"message": "completion of task!",
                "data": {
                    "download_status": True
                }}


class Download(Resource):
    """用户数据结果下载视图"""
    # 添加装饰器
    method_decorators = {
        "get": [login_required]
    }

    def get(self):
        # 1.获取用户id
        user_id = g.user_id
        # 2.调取数据库数据,查看是否服务器存在数据

        # 2.1 如果用户还为提交过参数,即数据库中还未有用户的参数配置
        # db_user_id = UserParameter.query.options(load_only(UserParameter.user_id)).filter(
        #     UserParameter.user_id == user_id).first()
        # if not db_user_id:
        #     return {"message": "用户还未提交任务"}, 401
        # # 2.2 如果用户的下载状态为Flase,即说明没有可下载文件.
        # download_status = UserParameter.query.options(load_only(UserParameter.download_status)).filter(
        #     UserParameter.user_id == user_id).first()
        # if not download_status:
        #     return {"message": "没有可下载的文件"}, 401
        # 2.3 如果status==True,则返回用户的结果下载地址
        # # 文件下载
        # 3.返回下载url
        try:
            dowmload_file = send_from_directory(f"../user_file", f"{user_id}.zip", as_attachment=True)
        except:
            dowmload_file = None
            abort(400, "No downloadable files found!")

        return dowmload_file
        # return {"message": "OK"}


def process_validation(modul1, modul2, modul3, modul4, modul5):
    print("流程选择:", modul1, modul2, modul3, modul4, modul5)
    if modul2 == 3 and (modul1 + modul3 + modul4 + modul5):
        abort(400, "In the current version, the Monte Carlo method can only be used alone.")
    # if (modul1 and not (modul2 + modul3 + modul4 + modul5)) or (  # 1
    #         modul2 and not (modul1 + modul3 + modul4 + modul5)) or (  # 2
    #         modul3 and not (modul2 + modul1 + modul4 + modul5)) or (  # 3
    #         modul4 and not (modul2 + modul3 + modul1 + modul5)) or (  # 4
    #         modul5 and not (modul2 + modul3 + modul4 + modul1)) or (  # 5
    #         modul1 and modul2 and not (modul3 + modul4 + modul5)) or (  # 12
    #         modul2 and modul3 and not (modul1 + modul4 + modul5)) or (  # 23
    #         modul1 and modul3 and not (modul2 + modul4 + modul5)) or (  # 13
    #         modul3 and modul4 and not (modul1 + modul2 + modul5)) or (  # 34
    #         modul4 and modul5 and not (modul3 + modul1 + modul2)) or (  # 45
    #         modul1 and modul2 and modul3 and not (modul4 + modul5)) or (  # 123
    #         modul3 and modul4 and modul5 and not (modul1 + modul2)) or (  # 345
    #         modul1 >= 10 and modul3 < 10):  # modul1>=10情况
    #     pass
    # else:
    #     abort(400, "Process configuration error. Please reconfigure.")


if __name__ == '__main__':
    pass
