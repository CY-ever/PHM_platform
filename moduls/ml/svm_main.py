import numpy as np
import pandas as pd
from flask import abort

from moduls.ml.dataset import import_data
import scipy.io
from moduls.ml.pso import PSO
from moduls.ml.sa import SA
from moduls.ml.svm import SVM_Model

from moduls.ml.utils import plot_confuse, plot_rul

"""
    1.为SA增加了结果图片保存路径参数
"""


def run_svm_pso(x_train, x_test, y_train, y_test, var_size, rul_pre=False,
                part_num=2, num_itr=5, output_image=0, save_path=None):
    """
    Main function for the SVM and PSO.

    :param path_to_data: string, Folder of the data files
    :param var_size: list, upper and under boundaries of all variables
    :param part_num: integer, number of particles
    :param num_itr: integer, number of iterations
    :param save_path: string, 图片保存路径
    :return: None
    """
    # x_train, x_test, y_train, y_test = import_data(path_to_data, model='SVM')
    svm = SVM_Model(x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test, rul_pre=rul_pre)
    # optimization=True)
    pso = PSO(svm.get_score, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
              rul_pre=rul_pre, part_num=part_num, num_itr=num_itr,
              var_size=var_size, net="SVM", output_image=output_image, save_path=save_path)
    # (objective, part_num, num_itr, var_size, candidate=None, net=None, save_path=None)
    param_dict = pso.run()
    return svm, param_dict


def run_svm_sa(x_train, x_test, y_train, y_test, var_size, rul_pre=False,
               initial_temp=500, final_temp=1,
               alpha=0.9, max_iter=5, output_image=0, save_path=None):
    """
    Main function for the SVM and SA.

    :param path_to_data: string, Folder of the data files
    :param var_size: list, upper and under boundaries of all variables
    :param initial_temp: double, manually set initial_temp, e.g. 500
    :param final_temp: double, stop_temp, e.g. 1
    :param alpha: double, temperature changing step, normal range[0.900, 0.999], e.g.0.9
    :param max_iter: int, maximal iteration number e.g. 30
    :return: None
    """
    # x_train, x_test, y_train, y_test = import_data(path_to_data, model='SVM')
    svm = SVM_Model(x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test, rul_pre=rul_pre)
    # optimization=True)
    # svm.get_score()
    sa = SA(svm.get_score, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
            rul_pre=rul_pre, initial_temp=initial_temp, final_temp=final_temp,
            alpha=alpha, max_iter=max_iter, var_size=var_size, net="SVM",
            output_image=output_image, save_path=save_path)
    # # (objective, initial_temp, final_temp, alpha, max_iter, var_size, net)
    param_dict = sa.run()
    return svm, param_dict


"""
SA优化器超参数/input:
PATH_TO_DATA:string/训练数据路径
VAR_SIZE:所有变量的上限和下限[[x1_min,x1_max], [x2_min,x2_max],..., [xn_min,xn_max]]
INITIAL_TEMP:float/default=500
FINAL_TEMP:float/default=1
ALPHA:float/学习率/衰减因子
MAX_ITER: int/最大迭代次数
NET:string/选择优化对象："DBN", "CNN", "SVM":通用，仅应用于结果打印，也可以是AN
"""


def run_svm_ga(x_train, x_test, y_train, y_test, var_size, rul_pre=False,
               threshold=100.9960, dna_size=9, pop_size=6,
               cross_rate=0.3, mutation_rate=0.1, n_generations=10,
               output_image=0, save_path=None):
    """
    Main function for the SVM and SA.

    :param path_to_data: string, Folder of the data files
    :param n_generations:
    :param mutation_rate:
    :param cross_rate:
    :param pop_size: int,迭代次数
    :param dna_size:
    :param threshold: float, 优化停止阙值
    :param save_path: string, 保存路径
    :return: None
    """
    # 1.数据准备
    # x_train, x_test, y_train, y_test = import_data(path_to_data, model='SVM')

    # 2.模型准备
    svm = SVM_Model(x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test, rul_pre=rul_pre)
    # 3.参数范围准备(已封装)
    # selection = [0, 100, 0, 0.01, pop_size, 209, 100, 155, 20, 20, 128, 128, 'default', 'default', 'mse', 30]

    # 4.开始优化
    from moduls.ml.ga import run
    best_generation, dna_set = run(svm, "SVM", var_size=var_size, x_train=x_train, x_test=x_test, y_train=y_train,
                                   y_test=y_test,
                                   rul_pre=rul_pre, threshold=threshold,
                                   dna_size=dna_size, pop_size=pop_size,
                                   cross_rate=cross_rate, mutation_rate=mutation_rate,
                                   n_generations=n_generations,
                                   output_image=output_image, save_path=save_path)

    # 5.优化结果参数提取
    param_dict = {"C": dna_set[0][0], "gamma": dna_set[0][1]}
    return svm, param_dict


"""


SVM优化器输出/output:
C = 25.100886524084405 
gamma = 0.006896530555596532

"""


def SVM(x_train, x_test, y_train, y_test, opt_option=None, rul_pre=False,
        output_image=0, C=95, gamma=0.01,
        pso_part_num=2, pso_num_itr=5, sa_initial_temp=500, sa_final_temp=1,
        sa_alpha=0.9, sa_max_iter=5,
        ga_threshold=100.9960, ga_dna_size=9, ga_pop_size=6, ga_cross_rate=0.3,
        ga_mutation_rate=0.1, ga_n_generations=10,
        save_path=None):
    """
    Main function to call the selected model and optimizer

    :param path_to_data: list, upper and under boundaries of all variables
    :param opt_option: string, 'PSO' or 'SA' 优化方法
    :param gamma: 惩罚系数
    :param C: 核函数系数
    :param pso_part_num: integer, number of particles, default:2
    :param pso_num_itr: integer, number of iterations, default:5
    :param sa_initial_temp: double, manually set initial_temp, e.g. 500
    :param sa_final_temp: double, stop_temp, e.g. 1
    :param sa_alpha: double, temperature changing step, normal range[0.900, 0.999], e.g.0.9
    :param sa_max_iter: int, maximal iteration number e.g. 30
    :param ga_n_generations:
    :param ga_mutation_rate:
    :param ga_cross_rate:
    :param ga_pop_size: int,迭代次数
    :param ga_dna_size:
    :param ga_threshold: float, 优化停止阙值
    :param save_path: string, 图片保存路径
    :return: None
    """
    # below should get from config
    # 参数校验
    if not opt_option:
        if C <= 0:
            abort(400, "ERROR: The parameter C must be a positive real number.")
        if gamma <= 0 or gamma >= 1:
            abort(400, "ERROR: The value range of parameter SA gamma is (0, 1).")

    var_size = [[15, 100], [0.001, 0.01]]  # var_size = [C,gamma]
    # x_train, x_test, y_train, y_test = import_data(path_to_data, model='SVM')

    if opt_option == 'PSO':
        svm, params_dict = run_svm_pso(x_train, x_test, y_train, y_test, var_size,
                                       rul_pre=rul_pre,
                                       part_num=pso_part_num, num_itr=pso_num_itr,
                                       output_image=output_image, save_path=save_path)
        # 使用优化后得参数训练模型，打印结果
        params = [params_dict["C"], params_dict["gamma"]]
        y_pre, score = svm.train_svm(params)
        if not rul_pre:
            plot_confuse(y_pre, y_test, output_image=output_image, save_path=save_path)
        else:
            plot_rul(y_pre, y_test, output_image=output_image, save_path=save_path)
        y_pre = np.reshape(y_pre, (-1, 1))
        params_dict["accuracy"] = score
        return y_pre, params_dict

    elif opt_option == 'SA':
        svm, params_dict = run_svm_sa(x_train, x_test, y_train, y_test, var_size,
                                      rul_pre=rul_pre,
                                      initial_temp=sa_initial_temp, final_temp=sa_final_temp,
                                      alpha=sa_alpha, max_iter=sa_max_iter,
                                      output_image=output_image, save_path=save_path)
        # 使用优化后得参数训练模型，打印结果
        params = [params_dict["C"], params_dict["gamma"]]
        y_pre, score = svm.train_svm(params)
        if not rul_pre:
            plot_confuse(y_pre, y_test, output_image=output_image, save_path=save_path)
        else:
            plot_rul(y_pre, y_test, output_image=output_image, save_path=save_path)
        y_pre = np.reshape(y_pre, (-1, 1))
        params_dict["accuracy"] = score
        return y_pre, params_dict

    elif opt_option == 'GA':
        # from ga import run_svm_ga
        svm, params_dict = run_svm_ga(x_train, x_test, y_train, y_test, var_size=var_size, threshold=ga_threshold,
                                      dna_size=ga_dna_size,
                                      rul_pre=rul_pre,
                                      pop_size=ga_pop_size, cross_rate=ga_cross_rate,
                                      mutation_rate=ga_mutation_rate, n_generations=ga_n_generations,
                                      output_image=output_image, save_path=save_path)
        # 使用优化后得参数训练模型，打印结果
        params = [params_dict["C"], params_dict["gamma"]]
        y_pre, score = svm.train_svm(params)
        if not rul_pre:
            plot_confuse(y_pre, y_test, output_image=output_image, save_path=save_path)
        else:
            plot_rul(y_pre, y_test, output_image=output_image, save_path=save_path)
        y_pre = np.reshape(y_pre, (-1, 1))

        params_dict["accuracy"] = score
        return y_pre, params_dict

    elif opt_option is None:
        # 如果不指定优化种类，则直接使用默认参数训练模型
        svm = SVM_Model(x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test,
                        rul_pre=rul_pre)
        params = [C, gamma]
        y_pre, score = svm.train_svm(params)
        if not rul_pre:
            plot_confuse(y_pre, y_test, output_image=output_image, save_path=save_path)
        else:
            plot_rul(y_pre, y_test, output_image=output_image, save_path=save_path)

        y_pre = np.reshape(y_pre, (-1, 1))
        params_dict = {"C": params[0], "gamma": params[1]}
        params_dict["accuracy"] = score
        return y_pre, params_dict

    else:
        print("without this optimization option")
        raise


if __name__ == '__main__':
    option = "GA"  # switch between "SA", "PSO", "GA"，None
    data_path = './Dataset/'
    save_path = './test/'

    # x_train, x_test, y_train, y_test = import_data(data_path, model='SVM')

    from test import read_data_csv

    data_path = "./Dataset/feature_1_1_all.csv"
    x_train, x_test, y_train, y_test = read_data_csv(data_path)
    size_train = x_train.shape[0]
    size_test = x_test.shape[0]
    y_train = y_train.reshape(size_train, 1)
    y_test = y_test.reshape(size_test, 1)

    # scipy.io.savemat('traindata_rul.mat', {'train_data': x_train,})
    # scipy.io.savemat('testdata_rul.mat', {'test_data': x_test, })
    # scipy.io.savemat('trainlabel_rul.mat', {'train_label': y_train, })
    # scipy.io.savemat('testlabel_rul.mat', {'train_label': y_test, })

    """
        rul_pre:
            input:
            x(2327,38)
            y(2327,1/4)
    """

    acc = SVM(x_train, x_test, y_train, y_test, opt_option=option, rul_pre=True,
              output_image=0, save_path=save_path)
    # print("最后分数为：", acc)

    # 数据集信息
    """
    Y: [0, 1, 2, 3]
    X_train.shape: [11821, 576]
    X_test.shape: [2956, 576]
    Test: 20%
    """
    # x_train_all, x_test_all, y_train_all, y_test_all = import_data(data_path)
    # print(y_train_all)
