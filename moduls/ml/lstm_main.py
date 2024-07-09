# import keras
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
import numpy as np
from flask import abort

from moduls.ml.dataset import import_data
from moduls.ml.ga import run
from moduls.ml.pso import PSO
from moduls.ml.py.Lstm import get_accuracy, train_model
from moduls.ml.sa import SA
from moduls.ml.utils import plot_confuse, plot_rul


def run_lstm_ga(x_train, x_test, y_train, y_test, rul_pre=False, var_size=None,
                threshold=100.9960, dna_size=9, pop_size=6,
                cross_rate=0.3, mutation_rate=0.1, n_generations=10,
                output_image=0, save_path=None):
    """

    :return:
    """
    # 1. 数据准备
    # path_to_data = './Dataset/'
    # save_path = './'
    # x_train, x_test, y_train, y_test = import_data(path_to_data, model='AE')

    """
    # results[LayerCount, units1, units2, units3, epochs, batchSize, denseActivation, optimizer, loss]
    # LayerCount[1, 3], int
    # units1[210, 250], int
    # units2[160, 209], int
    # units3[100, 155], int
    # epochs[20, 20], int
    # batchSize[128, 128], int
    # denseActivation['softmax', 'sigmoid', 'relu', 'tanh']
    # optimizer['adam', 'rmsprop', 'adagrad']
    # loss['mse', 'mae', 'categorical_crossentropy']
    """

    # 4.开始优化

    best_generation, dna_set = run(get_accuracy, "LSTM", rul_pre=rul_pre,
                                   var_size=var_size, output_image=output_image,
                                   x_train=x_train, x_test=x_test,
                                   y_train=y_train, y_test=y_test,
                                   save_path=save_path,
                                   threshold=threshold,
                                   dna_size=dna_size, pop_size=pop_size,
                                   cross_rate=cross_rate, mutation_rate=mutation_rate,
                                   n_generations=n_generations)

    # 5.优化结果参数提取
    # results[LayerCount, units1, units2, units3, epochs, batchSize, denseActivation, optimizer, loss]
    param_dict = {"lstm_count": dna_set[0][0],
                  "units1": dna_set[0][1],
                  "units2": dna_set[0][2],
                  "units3": dna_set[0][3],
                  "dropoutRate": dna_set[0][4],
                  "epochs": dna_set[0][5],
                  "batchSize": dna_set[0][6],
                  "denseActivation": dna_set[0][7],
                  "optimizer": dna_set[0][8],
                  "loss": dna_set[0][9],
                  }

    return param_dict


def run_lstm_sa(x_train, x_test, y_train, y_test, var_size, rul_pre=False,
                output_image=0, initial_temp=500, final_temp=1,
                alpha=0.9, max_iter=2, save_path="./"):
    """
    def run_svm_sa(path_to_data, var_size, initial_temp=500, final_temp=1,
               alpha=0.9, max_iter=2, save_path=None):
    :return:
    """

    sa = SA(get_accuracy, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
            rul_pre=rul_pre, output_image=output_image,
            initial_temp=initial_temp, final_temp=final_temp,
            alpha=alpha, max_iter=max_iter, var_size=var_size, net="LSTM",
            save_path=save_path)
    # # (objective, initial_temp, final_temp, alpha, max_iter, var_size, net)

    # 5.优化结果参数提取
    # results[LayerCount, units1, units2, units3, epochs, batchSize, denseActivation, optimizer, loss]
    param_dict = sa.run()

    return param_dict


def run_lstm_pso(x_train, x_test, y_train, y_test, var_size,
                 part_num=2, num_itr=5, rul_pre=False,
                 output_image=0, save_path=None):
    """
    def run_svm_sa(path_to_data, var_size, initial_temp=500, final_temp=1,
               alpha=0.9, max_iter=2, save_path=None):
    :return:
    """
    # 1. 数据准备
    # path_to_data = './Dataset/'
    # save_path = './'
    # x_train, x_test, y_train, y_test = import_data(path_to_data, model='AE')

    """
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

    # 4.开始优化

    pso = PSO(get_accuracy, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
              rul_pre=rul_pre, part_num=part_num, num_itr=num_itr,
              var_size=var_size, net="LSTM", output_image=output_image, save_path=save_path)
    # # (objective, initial_temp, final_temp, alpha, max_iter, var_size, net)

    # 5.优化结果参数提取
    # results[LayerCount, units1, units2, units3, epochs, batchSize, denseActivation, optimizer, loss]
    param_dict = pso.run()

    return param_dict


def LSTM(x_train, x_test, y_train, y_test, opt_option=None, rul_pre=False,
         output_image=0,
         layer_count=2, units1=150, units2=160,
         units3=120, dropoutRate=0.2, epochs=20, batchSize=128, denseActivation='sigmoid',
         optimizer='adam',
         pso_part_num=2, pso_num_itr=5, sa_initial_temp=500, sa_final_temp=1,
         sa_alpha=0.9, sa_max_iter=10,
         ga_threshold=100.9960, ga_dna_size=9, ga_pop_size=6, ga_cross_rate=0.3,
         ga_mutation_rate=0.1, ga_n_generations=10,
         save_path=None):
    """
    Main function to call the selected model and optimizer

    :param path_to_data: list, upper and under boundaries of all variables
    :param opt_option: string, 'PSO' or 'SA' or 'GA'优化方法
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
        print("layer_count", layer_count, type(layer_count))
        if layer_count > 3 or layer_count < 1:
            abort(400, "ERROR: The parameter layer count must take values in [1, 2, 3].")
        if units1 < 1:
            abort(400, "ERROR: The parameter units1 must be a positive integer.")
        if units2 < 1:
            abort(400, "ERROR: The parameter units2 must be a positive integer.")
        if units3 < 1:
            abort(400, "ERROR: The parameter units3 must be a positive integer.")
        if epochs < 2:
            abort(400, "ERROR: The parameter epochs must be a positive integer and greater than 1.")
        if batchSize < 1:
            abort(400, "ERROR: The parameter batch size must be a positive integer.")
        if dropoutRate >= 1 or dropoutRate <= 0:
            abort(400, "ERROR: The value range of parameter dropout rate is (0, 1).")

    # below should get from config
    # size_train = x_train.shape[0]
    # size_test = x_test.shape[0]
    # x_train = x_train.reshape(size_train, 1, x_train.shape[1])
    # x_test = x_test.reshape(size_test, 1, x_test.shape[1])
    #
    # y_train_max = int(max(y_train))+1
    # y_test_max = int(max(y_test))+1
    # # print("y_train.max:", y_train_max)
    # from moduls.ml.dataset import to_cat
    # y_train = to_cat(y_train, num_classes=y_train_max)
    # y_test = to_cat(y_test, num_classes=y_test_max)

    # 参数校验:

    # 定义优化参数范围
    var_size = [[1, 3],
                [100, 200],
                [100, 200],
                [150, 200],
                [0.1, 0.3],
                [20, 20],
                [128, 128],
                ['softmax', 'sigmoid', 'relu', 'tanh'],
                ['adam', 'rmsprop', 'adagrad'],
                ['mse', 'mse']]
    # loss = keras.losses.mse
    if not rul_pre:
        loss = 'categorical_crossentropy'
        var_size[9] = ['categorical_crossentropy', 'categorical_crossentropy']

    else:
        loss = 'mse'
        var_size[9] = ['mse', 'mse']
    # results = [2, 220, 200, 120, 2, 128, 'relu', 'adam', 'mse'/'mae'/'categorical_crossentropy']
    results = [layer_count, units1, units2, units3, dropoutRate, epochs, batchSize, denseActivation, optimizer, loss]

    if opt_option == 'PSO':
        params_dict = run_lstm_pso(x_train, x_test, y_train, y_test, var_size, rul_pre=rul_pre,
                                   part_num=pso_part_num, num_itr=pso_num_itr,
                                   output_image=output_image, save_path=save_path)
        # 使用优化后得参数训练模型，打印结果
        params = [params_dict["lstm_count"], params_dict["units1"],
                  params_dict["units2"], params_dict["units3"],
                  params_dict["dropoutRate"],
                  params_dict["epochs"], params_dict["batchSize"],
                  params_dict["denseActivation"], params_dict["optimizer"],
                  params_dict["loss"]]
        y, acc = get_accuracy(x_train, x_test, y_train, y_test, params, rul_pre=rul_pre)
        if not rul_pre:
            y_test_use = y[0]
            y_pre = y[1]
            plot_confuse(y_pre, y_test_use, output_image=output_image, save_path=save_path)
        else:
            y_test_use = y[0]
            y_pre = y[1]
            plot_rul(y_pre, y_test_use, output_image=output_image, save_path=save_path)
        # return acc
        y_pre = y[1]
        y_pre = np.reshape(y_pre, (-1, 1))
        params_dict["accuracy"] = acc
        return y_pre, params_dict

    elif opt_option == 'SA':
        params_dict = run_lstm_sa(x_train, x_test, y_train, y_test, var_size, rul_pre=rul_pre,
                                  initial_temp=sa_initial_temp, final_temp=sa_final_temp,
                                  alpha=sa_alpha, max_iter=sa_max_iter,
                                  output_image=output_image, save_path=save_path)
        # 使用优化后得参数训练模型，打印结果
        params = [params_dict["lstm_count"], params_dict["units1"],
                  params_dict["units2"], params_dict["units3"],
                  params_dict["dropoutRate"],
                  params_dict["epochs"], params_dict["batchSize"],
                  params_dict["denseActivation"], params_dict["optimizer"],
                  params_dict["loss"]]
        y, acc = get_accuracy(x_train, x_test, y_train, y_test, params, rul_pre=rul_pre)
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
        print("rul_pre", rul_pre)
        params_dict = run_lstm_ga(x_train, x_test, y_train, y_test,  # results,
                                  rul_pre=rul_pre, var_size=var_size,
                                  threshold=ga_threshold, dna_size=ga_dna_size,
                                  pop_size=ga_pop_size, cross_rate=ga_cross_rate,
                                  mutation_rate=ga_mutation_rate, n_generations=ga_n_generations,
                                  output_image=output_image, save_path=save_path)
        # 用优化后得参数训练模型，打印结果
        params = [params_dict["lstm_count"], params_dict["units1"],
                  params_dict["units2"], params_dict["units3"],
                  params_dict["dropoutRate"],
                  params_dict["epochs"], params_dict["batchSize"],
                  params_dict["denseActivation"], params_dict["optimizer"],
                  params_dict["loss"]]

        y, acc = get_accuracy(x_train, x_test, y_train, y_test, params, rul_pre=rul_pre)
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
        # x_train_all, x_test_all, y_train, y_test

        print("results", results)
        print("Dense Activation:", results[7])
        y, acc = get_accuracy(x_train, x_test, y_train, y_test, results, rul_pre=rul_pre)

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

        params_dict = {"lstm_count": results[0], "units1": results[1], "units2": results[2],
                       "units3": results[3], "dropoutRate": results[4], "epochs": results[5],
                       "batchSize": results[6], "denseActivation": results[7], "optimizer": results[8],
                       "loss": results[9]}
        params_dict["accuracy"] = acc
        return y_pre, params_dict

    else:
        print("without this optimization option")
        raise


if __name__ == '__main__':
    data_path = './Dataset/'
    save_path = './test/'
    opt_option = "PSO"

    x_train, x_test, y_train, y_test = import_data(data_path, model='LSTM')

    from test import read_data_csv

    data_path = "./Dataset/feature_1_1_all.csv"
    x_train, x_test, y_train, y_test = read_data_csv(data_path)
    size_train = x_train.shape[0]
    size_test = x_test.shape[0]
    x_train = x_train.reshape(size_train, x_train.shape[1], 1)
    x_test = x_test.reshape(size_test, x_test.shape[1], 1)
    y_train = y_train.reshape(size_train, 1)
    y_test = y_test.reshape(size_test, 1)

    """
    rul_pre:
        input:
        x(2327,38,1)
        y(2327,1)
    """
    res = LSTM(x_train, x_test, y_train, y_test, output_image=0,
               rul_pre=True, opt_option=opt_option, save_path=save_path)
    print("最终分数为：", res)

    """
       # selection=[1, 4, 100,200,100,200,150,200,0.1,0.3,20,20,128,128,'default','default','categorical_crossentropy',30] 
       # results[LSTMCount, units1, units2, units3, dropoutRate, epochs, batchSize, denseActivation, optimizer, loss]
       # LSTMCount[1, 4], int
       # units1[100, 200], int
       # units2[100, 200], int
       # units3[150, 200], int
       # dropoutRate[0.1, 0.3]
       # epochs[20, 20], int
       # batchSize[128, 128], int
       # denseActivation['softmax', 'sigmoid', 'relu', 'tanh']
       # optimizer['adam', 'rmsprop', 'adagrad']
       # loss['mse', 'mae', 'categorical_crossentropy']
    """
    results = [2, 220, 200, 120, 2, 128, 'relu', 'adam', 'mae']
