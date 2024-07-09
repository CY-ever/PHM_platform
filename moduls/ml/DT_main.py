import numpy as np
from flask import abort

from moduls.ml.dataset import import_data
from moduls.ml.pso import PSO
from moduls.ml.py.Decisiontree import get_score
from moduls.ml.sa import SA
from moduls.ml.utils import plot_confuse, plot_rul


def run_dt_pso(x_train, x_test, y_train, y_test, var_size, rul_pre=False,
               output_image=0, part_num=2, num_itr=5, save_path=None):
    """
    Main function for the DBN and PSO.

    :param path_to_data: string, Folder of the data files
    :param var_size: list, upper and under boundaries of all variables
    :return: None
    """

    pso = PSO(get_score, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, part_num=part_num,
              num_itr=num_itr,
              rul_pre=rul_pre, output_image=output_image,
              var_size=var_size, net="DT", save_path=save_path)
    param_dict = pso.run()
    param_dict["best"] = 1 - pso.GlobalBest_Cost
    return param_dict


def run_dt_sa(x_train, x_test, y_train, y_test, var_size, rul_pre=False,
              output_image=0, initial_temp=500, final_temp=1,
              alpha=0.9, max_iter=2, save_path=None):
    """
    Main function for the DBN and SA.

    :param path_to_data: string, Folder of the data files
    :param var_size: list, upper and under boundaries of all variables
    :return: None
    """

    sa = SA(get_score, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
            rul_pre=rul_pre, output_image=output_image,
            initial_temp=initial_temp, final_temp=final_temp,
            alpha=alpha, max_iter=max_iter, var_size=var_size, net="DT",
            save_path=save_path)
    # objective, initial_temp, final_temp, alpha, max_iter, var_size, candidate, net
    param_dict = sa.run()
    param_dict["best"] = 1 - min(sa.costs)
    return param_dict


def run_dt_ga(x_train, x_test, y_train, y_test, var_size, rul_pre=False,
              output_image=0, threshold=100.9960, dna_size=9, pop_size=6,
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
    # x_train, x_test, y_train, y_test = import_data(path_to_data, model="DT")

    # 3.开始优化
    from moduls.ml.ga import run
    best_generation, dna_set = run(get_score, x_train=x_train, x_test=x_test,
                                   y_train=y_train, y_test=y_test, var_size=var_size,
                                   rul_pre=rul_pre, output_image=output_image,
                                   Model_name="DT", save_path=save_path, threshold=threshold,
                                   dna_size=dna_size, pop_size=pop_size,
                                   cross_rate=cross_rate, mutation_rate=mutation_rate,
                                   n_generations=n_generations)

    # 4.优化结果参数提取
    param_dict = {"max_depth": dna_set[0][0],
                  "max_leaf_nodes": dna_set[0][1],
                  "best": best_generation}
    return param_dict


# var_size = [[4, 20], [1, 12]]
def DT(x_train, x_test, y_train, y_test, opt_option=None, rul_pre=False,
       output_image=0, max_depth=20, max_leaf_nodes=600,
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
    # 参数校验
    if not opt_option:
        if max_depth < 1:
            abort(400, "ERROR: The parameter max depth must be a positive integer .")
        if max_leaf_nodes < 2 :
            abort(400, "ERROR: The parameter max leaf nodes must be a positive integer and greater than 2.")

    # below should get from config
    # var_size = [max_depth, max_leaf_nodes]
    var_size = [[4, 20], [2, 1000]]


    # x_train, x_test, y_train, y_test = import_data(path_to_data, model='RF')

    if opt_option == 'PSO':
        params_dict = run_dt_pso(x_train, x_test, y_train, y_test, var_size,
                                 rul_pre=rul_pre, output_image=output_image,
                                 part_num=pso_part_num, num_itr=pso_num_itr,
                                 save_path=save_path)
        # 使用优化后得参数训练模型，打印结果
        params = [params_dict["max_depth"],
                  params_dict["max_leaf_nodes"]]

        if not rul_pre:
            pre, score = get_score(x_train, x_test, y_train, y_test, params, rul_pre=rul_pre)
            # score = 1 - Error
            y_test_use = y_test.ravel()
            plot_confuse(pre, y_test_use, output_image=output_image, save_path=save_path)

        else:
            pre, score = get_score(x_train, x_test, y_train, y_test, params, rul_pre=rul_pre)
            y_test_use = y_test.ravel()
            plot_rul(pre, y_test_use, output_image=output_image, save_path=save_path)

            # score = 1 - Error

        y_pre = np.reshape(pre, (-1, 1))
        params_dict["accuracy"] = score
        return y_pre, params_dict

    elif opt_option == 'SA':
        params_dict = run_dt_sa(x_train, x_test, y_train, y_test, var_size,
                                rul_pre=rul_pre, output_image=output_image,
                                initial_temp=sa_initial_temp, final_temp=sa_final_temp,
                                alpha=sa_alpha, max_iter=sa_max_iter,
                                save_path=save_path)
        # 使用优化后得参数训练模型，打印结果
        params = [params_dict["max_depth"],
                  params_dict["max_leaf_nodes"]]
        if not rul_pre:
            pre, score = get_score(x_train, x_test, y_train, y_test, params, rul_pre=rul_pre)
            # score = 1 - Error
            y_test_use = y_test.ravel()
            plot_confuse(pre, y_test_use, output_image=output_image, save_path=save_path)

        else:
            pre, score = get_score(x_train, x_test, y_train, y_test, params, rul_pre=rul_pre)
            y_test_use = y_test.ravel()
            plot_rul(pre, y_test_use, output_image=output_image, save_path=save_path)
            # score = 1 - Error

        y_pre = np.reshape(pre, (-1, 1))
        params_dict["accuracy"] = score
        return y_pre, params_dict

    elif opt_option == 'GA':
        params_dict = run_dt_ga(x_train, x_test, y_train, y_test, var_size=var_size,
                                rul_pre=rul_pre, output_image=output_image,
                                threshold=ga_threshold, dna_size=ga_dna_size,
                                pop_size=ga_pop_size, cross_rate=ga_cross_rate,
                                mutation_rate=ga_mutation_rate, n_generations=ga_n_generations,
                                save_path=save_path)
        # 使用优化后得参数训练模型，打印结果
        params = [params_dict["max_depth"],
                  params_dict["max_leaf_nodes"]]

        if not rul_pre:
            pre, score = get_score(x_train, x_test, y_train, y_test, params, rul_pre=rul_pre)
            # score = 1 - Error
            y_test_use = y_test.ravel()
            plot_confuse(pre, y_test_use, output_image=output_image, save_path=save_path)

        else:
            pre, score = get_score(x_train, x_test, y_train, y_test, params, rul_pre=rul_pre)
            y_test_use = y_test.ravel()
            plot_rul(pre, y_test_use, output_image=output_image, save_path=save_path)
            # score = 1 - Error

        y_pre = np.reshape(pre, (-1, 1))
        params_dict["accuracy"] = score
        return y_pre, params_dict

    elif opt_option is None:
        # 如果不指定优化种类，则直接使用默认参数训练模型

        params = [max_depth, max_leaf_nodes]

        if not rul_pre:
            pre, score = get_score(x_train, x_test, y_train, y_test, params, rul_pre=rul_pre)
            # score = 1 - Error
            y_test_use = y_test.ravel()
            plot_confuse(pre, y_test_use, output_image=output_image, save_path=save_path)

        else:
            pre, score = get_score(x_train, x_test, y_train, y_test, params, rul_pre=rul_pre)
            y_test_use = y_test.ravel()
            plot_rul(pre, y_test_use, output_image=output_image, save_path=save_path)
            # score = 1 - Error

        y_pre = np.reshape(pre, (-1, 1))
        params_dict = {"max_depth": params[0], "max_leaf_nodes": params[1]}
        params_dict["accuracy"] = score
        return y_pre, params_dict



    else:
        print("without this optimization option")
        raise


if __name__ == '__main__':
    """
    Main function to call the selected model and optimizer
    """
    # Main function
    option = "PSO"  # switch between "SA", "PSO", "GA", None
    data_path = './Dataset/'
    # save_path = "./test"

    # x_train, x_test, y_train, y_test = import_data(data_path, model='DT')

    from test import read_data_csv

    data_path = "./Dataset/feature_1_1_all.csv"
    x_train, x_test, y_train, y_test = read_data_csv(data_path)
    size_train = x_train.shape[0]
    size_test = x_test.shape[0]
    y_train = y_train.reshape(size_train, 1)
    y_test = y_test.reshape(size_test, 1)

    """
        input_shape:
        x:(n_sample, features)(2327, 38)
        y:(n_sample, labels)(2327, 1)
    """

    acc = DT(x_train, x_test, y_train, y_test, rul_pre=True, output_image=0,
             pso_num_itr=5, pso_part_num=10, opt_option=option)
    print("最终准确率为：", acc)

    # results = [max_depth, max_leaf_nodes]
    selection = [4, 20, 1, 12]
