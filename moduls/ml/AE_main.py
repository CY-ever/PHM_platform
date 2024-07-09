from moduls.ml.dataset import import_data
from moduls.ml.ga import run
from moduls.ml.pso import PSO
from moduls.ml.py.Autoencoder import get_accuracy, train_model
from moduls.ml.sa import SA
from moduls.ml.utils import plot_confuse, plot_rul
import numpy as np
from flask import abort


def run_ae_ga(x_train, x_test, y_train, y_test, var_size, rul_pre=False,
              output_image=0,
              threshold=100.9960, dna_size=9, pop_size=6,
              cross_rate=0.3, mutation_rate=0.1, n_generations=10, save_path="./"):
    """

    :return:
    """

    # 4.开始优化
    best_generation, dna_set = run(get_accuracy, "AE", var_size=var_size,
                                   save_path=save_path, output_image=output_image,
                                   rul_pre=rul_pre, threshold=threshold,
                                   dna_size=dna_size, pop_size=pop_size,
                                   cross_rate=cross_rate, mutation_rate=mutation_rate,
                                   n_generations=n_generations,
                                   x_train=x_train, x_test=x_test,
                                   y_train=y_train, y_test=y_test)

    # 5.优化结果参数提取
    # results[LayerCount, units1, units2, units3, epochs, batchSize, denseActivation, optimizer, loss]
    param_dict = {"LayerCount": dna_set[0][0],
                  "units1": dna_set[0][1],
                  "units2": dna_set[0][2],
                  "units3": dna_set[0][3],
                  "epochs": dna_set[0][4],
                  "batchSize": dna_set[0][5],
                  "denseActivation": dna_set[0][6],
                  "optimizer": dna_set[0][7],
                  "loss": dna_set[0][8],
                  }

    return param_dict


def run_ae_sa(x_train, x_test, y_train, y_test, var_size, rul_pre=False,
              output_image=0,
              initial_temp=500, final_temp=1,
              alpha=0.9, max_iter=2, save_path="./"):
    """
    def run_svm_sa(path_to_data, var_size, initial_temp=500, final_temp=1,
               alpha=0.9, max_iter=2, save_path=None):
    :return:
    """
    sa = SA(get_accuracy, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
            rul_pre=rul_pre, output_image=output_image,
            initial_temp=initial_temp, final_temp=final_temp, alpha=alpha, max_iter=max_iter,
            var_size=var_size,
            net="AE", save_path=save_path)
    param_dict = sa.run()

    return param_dict


def run_ae_pso(x_train, x_test, y_train, y_test, var_size, rul_pre=False,
               output_image=0,
               part_num=2, num_itr=5, save_path=None):
    """
    def run_svm_sa(path_to_data, var_size, initial_temp=500, final_temp=1,
               alpha=0.9, max_iter=2, save_path=None):
    :return:
    """

    pso = PSO(get_accuracy, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
              rul_pre=rul_pre, output_image=output_image,
              part_num=part_num, num_itr=num_itr,
              var_size=var_size, net="AE", save_path=save_path)
    param_dict = pso.run()

    return param_dict


def AE(x_train, x_test, y_train, y_test, opt_option=None, rul_pre=False,
       output_image=0, layer_count=2, units1=220, units2=200,
       units3=120, epochs=20, batchSize=128, denseActivation='relu',
       optimizer='adam',
       pso_part_num=2, pso_num_itr=5, sa_initial_temp=500, sa_final_temp=1,
       sa_alpha=0.9, sa_max_iter=10,
       ga_threshold=100.9960, ga_dna_size=9, ga_pop_size=6, ga_cross_rate=0.3,
       ga_mutation_rate=0.1, ga_n_generations=1,
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
    # 参数校验
    if not opt_option:
        if layer_count not in [1, 2, 3]:
            abort(400, "ERROR: The parameter layer count must take values in [1, 2, 3].")
        if units1 < 1:
            abort(400, "ERROR: The parameter units1 must be a positive integer.")
        if units2 < 1:
            abort(400, "ERROR: The parameter units2 must be a positive integer.")
        if units3 < 1:
            abort(400, "ERROR: The parameter units3 must be a positive integer.")
        if epochs < 2:
            abort(400, "ERROR: The parameter epochs must be a positive integer and greater than 2.")
        if batchSize < 2:
            abort(400, "ERROR: The parameter batch size must be a positive integer and greater than 2.")

    var_size = [[1, 4],
                [210, 250],
                [160, 209],
                [100, 155],
                [20, 50],
                [128, 128],
                ['relu', 'relu'],
                ['adam', 'adam'],
                ['mse', 'mse']]
    # x_train, x_test, y_train, y_test = import_data(path_to_data, model='AE')

    # results = [2, 220, 200, 120, 2, 128, 'relu', 'adam', 'mse']

    if not rul_pre:
        # denseActivation='softmax'
        # var_size[6] = ['softmax', 'softmax']
        var_size[8] = ['categorical_crossentropy', 'categorical_crossentropy']
    else:
        # denseActivation = 'sigmoid'
        # var_size[6] = ['sigmoid', 'sigmoid']
        var_size[8] = ['mse', 'mse']
    results = [layer_count, units1, units2, units3, epochs, batchSize, denseActivation, optimizer, 'mse']

    if opt_option == 'PSO':
        params_dict = run_ae_pso(x_train, x_test, y_train, y_test, var_size,
                                 rul_pre=rul_pre, output_image=output_image,
                                 part_num=pso_part_num, num_itr=pso_num_itr,
                                 save_path=save_path)
        # 使用优化后得参数训练模型，打印结果
        params = [params_dict["LayerCount"], params_dict["units1"],
                  params_dict["units2"], params_dict["units3"],
                  params_dict["epochs"], params_dict["batchSize"],
                  params_dict["denseActivation"], params_dict["optimizer"],
                  params_dict["loss"]]

        y, error = get_accuracy(x_train, x_test, y_train, y_test,
                                params, rul_pre=rul_pre)
        acc = 1 - error
        if not rul_pre:
            y_test_use = y[0]
            y_pre = y[1]
            plot_confuse(y_pre, y_test_use, output_image=output_image, save_path=save_path)
        else:
            y_test_use = y[0]
            y_pre = y[1]
            plot_rul(y_pre, y_test_use, output_image=output_image, save_path=save_path)

        y_pre = y[1]
        y_pre = np.reshape(y_pre, (-1, 1))
        params_dict["accuracy"] = acc
        return y_pre, params_dict


    elif opt_option == 'SA':
        params_dict = run_ae_sa(x_train, x_test, y_train, y_test, var_size,
                                rul_pre=rul_pre, output_image=output_image,
                                initial_temp=sa_initial_temp, final_temp=sa_final_temp,
                                alpha=sa_alpha, max_iter=sa_max_iter,
                                save_path=save_path)
        # 使用优化后得参数训练模型，打印结果
        params = [params_dict["LayerCount"], params_dict["units1"],
                  params_dict["units2"], params_dict["units3"],
                  params_dict["epochs"], params_dict["batchSize"],
                  params_dict["denseActivation"], params_dict["optimizer"],
                  params_dict["loss"]]

        y, error = get_accuracy(x_train, x_test, y_train, y_test,
                                     params, rul_pre=rul_pre)
        acc = 1 - error
        if not rul_pre:
            y_test_use = y[0]
            y_pre = y[1]
            plot_confuse(y_pre, y_test_use, output_image=output_image, save_path=save_path)
        else:
            y_test_use = y[0]
            y_pre = y[1]
            plot_rul(y_pre, y_test_use, output_image=output_image, save_path=save_path)
        y_pre = y[1]
        y_pre = np.reshape(y_pre, (-1, 1))
        params_dict["accuracy"] = acc
        return y_pre, params_dict

    elif opt_option == 'GA':
        params_dict = run_ae_ga(x_train, x_test, y_train, y_test, var_size,
                                rul_pre=rul_pre, output_image=output_image,
                                threshold=ga_threshold, dna_size=ga_dna_size,
                                pop_size=ga_pop_size, cross_rate=ga_cross_rate,
                                mutation_rate=ga_mutation_rate, n_generations=ga_n_generations,
                                save_path=save_path)
        # 用优化后得参数训练模型，打印结果
        params = [params_dict["LayerCount"], params_dict["units1"],
                  params_dict["units2"], params_dict["units3"],
                  params_dict["epochs"], params_dict["batchSize"],
                  params_dict["denseActivation"], params_dict["optimizer"],
                  params_dict["loss"]]

        y, error = get_accuracy(x_train, x_test, y_train, y_test,
                                     params, rul_pre=rul_pre)
        acc = 1 - error
        if not rul_pre:
            y_test_use = y[0]
            y_pre = y[1]
            plot_confuse(y_pre, y_test_use, output_image=output_image, save_path=save_path)
        else:
            y_test_use = y[0]
            y_pre = y[1]
            plot_rul(y_pre, y_test_use, output_image=output_image, save_path=save_path)
        y_pre = y[1]
        y_pre = np.reshape(y_pre, (-1, 1))
        params_dict["accuracy"] = acc
        return y_pre, params_dict

    elif opt_option is None:
        # 如果不指定优化种类，则直接使用默认参数训练模型
        params = [layer_count, units1, units2, units3, epochs, batchSize, denseActivation, optimizer, "mae"]
        y, error = get_accuracy(x_train, x_test, y_train, y_test,
                                     params, rul_pre=rul_pre)
        acc = 1 - error
        if not rul_pre:
            y_test_use = y[0]
            y_pre = y[1]
            plot_confuse(y_pre, y_test_use, output_image=output_image, save_path=save_path)
        else:
            y_test_use = y[0]
            y_pre = y[1]
            plot_rul(y_pre, y_test_use, output_image=output_image, save_path=save_path)

        y_pre = y[1]
        y_pre = np.reshape(y_pre, (-1, 1))

        params_dict = {"LayerCount": params[0], "units1": params[1], "units2": params[2],
                       "units3": params[3], "epochs": params[4], "batchSize": params[5],
                       "denseActivation": params[6], "optimizer": params[7], "loss": params[8]}
        params_dict["accuracy"] = acc
        return y_pre, params_dict

    else:
        print("without this optimization option")
        raise


if __name__ == '__main__':
    data_path = './Dataset/'
    save_path = './'
    opt_option = "PSO"

    # x_train, x_test, y_train, y_test = import_data(data_path, model='AE')

    from test import read_data_csv

    data_path = "./Dataset/feature_1_1_all.csv"
    x_train, x_test, y_train, y_test = read_data_csv(data_path)
    size_train = x_train.shape[0]
    size_test = x_test.shape[0]
    y_train = y_train.reshape(size_train, 1)
    y_test = y_test.reshape(size_test, 1)

    """
        rul_pre:
            input:
            x(2327,38)
            y(2327,1/4)
    """

    res = AE(x_train, x_test, y_train, y_test, opt_option=opt_option, rul_pre=True,
             output_image=0, pso_part_num=5, epochs=100, save_path=None)
    print("最终分数为：", res)

    """
       # results = [2, 220, 200, 120, 2, 128, 'relu', 'adam', 'mse']
       # results[LayerCount, units1, units2, units3, epochs, batchSize, denseActivation, optimizer, loss]
       # LayerCount[1, 4], int
       # units1[210, 250], int
       # units2[160, 209], int
       # units3[100, 155], int
       # epochs[20, 20], int
       # batchSize[128, 128], int
       # denseActivation['softmax', 'sigmoid', 'relu', 'tanh']
       # optimizer['adam', 'rmsprop', 'adagrad']
       # loss['mse', 'mae', 'categorical_crossentropy']
    """
    # results = [2, 220, 200, 120, 2, 128, 'relu', 'adam', 'mae']
