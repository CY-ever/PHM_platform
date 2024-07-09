import numpy as np
import pandas as pd
from flask import abort

from moduls.ml.dataset import import_data
from moduls.ml.knn import KNN_Model
from moduls.ml.pso import PSO
from moduls.ml.sa import SA
from moduls.ml.utils import plot_confuse, plot_rul


def run_knn_pso(x_train, x_test, y_train, y_test, var_size,
                part_num=2, num_itr=5, rul_pre=False,
                output_image=0, save_path=None):
    """
    Main function for the SVM and PSO.

    :param path_to_data: string, Folder of the data files
    :param var_size: list, upper and under boundaries of all variables
    :param part_num: integer, number of particles
    :param num_itr: integer, number of iterations
    :param save_path: string, 图片保存路径
    :return: None
    """
    # x_train, x_test, y_train, y_test = import_data(path_to_data, model='KNN')
    knn = KNN_Model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    pso = PSO(knn.get_score, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
              part_num=part_num, num_itr=num_itr, rul_pre=rul_pre,
              var_size=var_size, net="KNN", output_image=output_image, save_path=save_path)
    # (objective, part_num, num_itr, var_size, candidate=None, net=None, save_path=None)
    param_dict = pso.run()
    return knn, param_dict


def run_knn_sa(x_train, x_test, y_train, y_test, var_size, rul_pre=False,
               initial_temp=500, final_temp=1,
               alpha=0.9, max_iter=2, output_image=0, save_path=None):
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
    # x_train, x_test, y_train, y_test = import_data(path_to_data, model='KNN')
    knn = KNN_Model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    sa = SA(knn.get_score, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
            rul_pre=rul_pre, initial_temp=initial_temp, final_temp=final_temp,
            alpha=alpha, max_iter=max_iter, var_size=var_size, net="KNN",
            output_image=output_image, save_path=save_path)
    # # (objective, initial_temp, final_temp, alpha, max_iter, var_size, net)
    param_dict = sa.run()
    return knn, param_dict


def run_knn_ga(x_train, x_test, y_train, y_test, var_size, rul_pre=False,
               output_image=0,
               threshold=100.9960, dna_size=9, pop_size=6,
               cross_rate=0.3, mutation_rate=0.1, n_generations=10, save_path="./"):
    """
    Main function for the KNN and SA.

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
    # x_train, x_test, y_train, y_test = import_data(path_to_data, model='KNN')

    # 2.模型准备
    knn = KNN_Model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    # 3.参数范围准备(已封装)
    # selection = [0, 100, 0, 0.01, pop_size, 209, 100, 155, 20, 20, 128, 128, 'default', 'default', 'mse', 30]

    # 4.开始优化
    from moduls.ml.ga import run
    best_generation, dna_set = run(knn, Model_name="KNN", rul_pre=rul_pre,
                                   output_image=output_image, var_size=var_size,
                                   save_path=save_path, threshold=threshold,
                                   dna_size=dna_size, pop_size=pop_size,
                                   cross_rate=cross_rate, mutation_rate=mutation_rate,
                                   n_generations=n_generations)

    # 5.优化结果参数提取
    param_dict = {"K": dna_set[0][0], "weights": dna_set[0][1]}
    return knn, param_dict


# def params_verification(x_train, x_test, pso_part_num, pso_num_itr, sa_max_iter, ga_pop_size):
#     from flask import abort
#     # 1. 输入数据形状校验
#     if len(x_train.shape) != 2 and len(x_test.shape) != 2:
#         abort(400, "x_train or x_test shape error!")
#     # 2.各参数校验
#     if pso_part_num < 2 or pso_part_num > 10:
#         abort(400, "pso_part_num parameter out of range[2, 10]!")
#     if pso_num_itr < 3 or pso_num_itr > 20:
#         abort(400, "pso_num_itr parameter out of range[3, 10]!")
#     if sa_max_iter < 3 or sa_max_iter > 20:
#         abort(400, "sa_max_iter parameter out of range[3, 10]!")
#     if ga_pop_size < 3 or ga_pop_size > 20:
#         abort(400, "ga_generations parameter out of range[3, 10]!")

def KNN(x_train, x_test, y_train, y_test, rul_pre=False, opt_option=None,
        output_image=0, K=5, weights="distance",
        pso_part_num=2, pso_num_itr=5, sa_initial_temp=500, sa_final_temp=1,
        sa_alpha=0.9, sa_max_iter=5,
        ga_threshold=100.9960, ga_dna_size=9, ga_pop_size=6, ga_cross_rate=0.3,
        ga_mutation_rate=0.1, ga_n_generations=2,
        save_path=None):
    """
    Main function to call the selected model and optimizer

    :param path_to_data: list, upper and under boundaries of all variables
    :param opt_option: string, 'PSO' or 'SA' 优化方法
    :param weights: weights
    :param K: number of neighbor
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

    # 参数验证
    if not opt_option:
        if K < 1:
            abort(400, "ERROR: The parameter K must be a positive integer.")

    var_size = [[1, 15], ["uniform", "distance"]]  # var_size = [K, weights]
    # x_train, x_test, y_train, y_test = import_data(path_to_data, model='KNN')

    if opt_option == 'PSO':
        knn, params_dict = run_knn_pso(x_train, x_test, y_train, y_test, var_size,
                                       rul_pre=rul_pre,
                                       part_num=pso_part_num, num_itr=pso_num_itr,
                                       output_image=output_image, save_path=save_path)
        # 使用优化后得参数训练模型，打印结果
        params = [params_dict["K"], params_dict["weights"]]
        y_pre, error = knn.get_score(params, rul_pre=rul_pre)
        accuracy = 1 - error
        if not rul_pre:
            plot_confuse(y_pre, y_test, output_image=output_image, save_path=save_path)
        else:
            plot_rul(y_pre, y_test, output_image=output_image, save_path=save_path)
        # 返回：准确率/损失
        y_pre = np.reshape(y_pre, (-1, 1))
        params_dict["accuracy"] = accuracy
        return y_pre, params_dict

    elif opt_option == 'SA':
        knn, params_dict = run_knn_sa(x_train, x_test, y_train, y_test, var_size,
                                      rul_pre=rul_pre,
                                      initial_temp=sa_initial_temp, final_temp=sa_final_temp,
                                      alpha=sa_alpha, max_iter=sa_max_iter,
                                      output_image=output_image, save_path=save_path)
        # 使用优化后得参数训练模型，打印结果
        params = [params_dict["K"], params_dict["weights"]]
        y_pre, error = knn.get_score(params, rul_pre=rul_pre)
        accuracy = 1 - error
        if not rul_pre:
            plot_confuse(y_pre, y_test, output_image=output_image, save_path=save_path)
        else:
            plot_rul(y_pre, y_test, output_image=output_image, save_path=save_path)
        # 返回：准确率/损失
        y_pre = np.reshape(y_pre, (-1, 1))
        params_dict["accuracy"] = accuracy
        return y_pre, params_dict

    elif opt_option == 'GA':
        knn, params_dict = run_knn_ga(x_train, x_test, y_train, y_test, rul_pre=rul_pre,
                                      var_size=var_size,
                                      threshold=ga_threshold, dna_size=ga_dna_size,
                                      pop_size=ga_pop_size, cross_rate=ga_cross_rate,
                                      mutation_rate=ga_mutation_rate, n_generations=ga_n_generations,
                                      output_image=output_image, save_path=save_path)
        # 使用优化后得参数训练模型，打印结果
        params = [params_dict["K"], params_dict["weights"]]
        y_pre, error = knn.get_score(params, rul_pre=rul_pre)
        accuracy = 1 - error
        if not rul_pre:
            plot_confuse(y_pre, y_test, output_image=output_image, save_path=save_path)
        else:
            plot_rul(y_pre, y_test, output_image=output_image, save_path=save_path)
        # 返回：准确率/损失
        y_pre = np.reshape(y_pre, (-1, 1))
        params_dict["accuracy"] = accuracy
        return y_pre, params_dict

    elif opt_option is None:
        # 如果不指定优化种类，则直接使用默认参数训练模型
        # x_train, x_test, y_train, y_test = import_data(path_to_data, model='KNN')
        knn = KNN_Model(x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test)
        params = [K, weights]
        y_pre, error = knn.get_score(params, rul_pre=rul_pre)

        accuracy = 1 - error
        if not rul_pre:
            plot_confuse(y_pre, y_test, output_image=output_image, save_path=save_path)
        else:
            plot_rul(y_pre, y_test, output_image=output_image, save_path=save_path)
        # 返回：准确率/损失

        # print("y_pre", y_pre.shape)
        y_pre = np.reshape(y_pre, (-1, 1))
        # print("y_pre", y_pre.shape)
        #
        params_dict = {"K":K, "weights":weights}
        params_dict["accuracy"] = accuracy
        return y_pre, params_dict


    else:
        print("without this optimization option")
        raise


if __name__ == '__main__':
    option = "GA"  # switch between "SA", "PSO", "GA", None
    data_path = './Dataset/'
    save_path = './test'

    """
    参数范围：
    k_range = range(1, 25)  # user input (under, upper bounds)
    weight_choices = ["uniform", "distance"]  # user input, string in list
    """
    # 故障分类
    x_train, x_test, y_train, y_test = import_data(data_path, model='KNN')
    # 寿命预测
    # from test import read_data_csv
    # data_path = "./Dataset/feature_1_1_all.csv"
    # x_train, x_test, y_train, y_test = read_data_csv(data_path)

    acc = KNN(x_train, x_test, y_train, y_test, output_image=3, rul_pre=False,
              opt_option=option, save_path=save_path)
    print("最终准确率为：", acc)
