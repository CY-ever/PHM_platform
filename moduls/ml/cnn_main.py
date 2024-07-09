from moduls.ml.dataset import import_data
import random
from moduls.ml.pso import PSO
from moduls.ml.cnn import CNN_Model
from moduls.ml.sa import SA
# import os
from flask import abort
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from moduls.ml.utils import plot_confuse, plot_rul
from moduls.utils.read_data import read_time_domain_signal
import numpy as np


def run_cnn_pso(x_train, x_test, y_train, y_test, var_size, discrete_candidate,
                rul_pre=False, output_image=0, epochs=5,
                part_num=2, num_itr=5, save_path=None):
    """
    Main function for the CNN and PSO.

    :param path_to_data: string, Folder of the data files
    :param var_size: list, upper and under boundaries of all variables
    :param discrete_candidate: list, list of discrete params, convolution layer and batch size
    :return: None
    """
    # create the CNN model
    # x_train, x_test, y_train, y_test = import_data(path_to_data, model='CNN')
    cnn = CNN_Model(x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test,
                    rul_pre=rul_pre,
                    discrete_candidate=discrete_candidate,
                    optimization=True,
                    epoch=epochs)
    pso = PSO(cnn.cnn_get_error, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
              part_num=part_num, rul_pre=rul_pre, output_image=output_image,
              num_itr=num_itr, var_size=var_size,
              candidate=discrete_candidate, net="CNN", save_path=save_path)
    params_dict = pso.run()

    return cnn, params_dict


def run_cnn_sa(x_train, x_test, y_train, y_test, var_size, discrete_candidate,
               rul_pre=False, output_image=0, epochs=5,
               initial_temp=500, final_temp=1, alpha=0.9, max_iter=10, save_path=None):
    """
    Main function for the CNN and SA.

    :param path_to_data: string, Folder of the data files
    :param var_size: list, upper and under boundaries of all variables
    :param discrete_candidate: list, list of discrete params, convolution layer and batch size
    :return: None
    """
    # x_train, x_test, y_train, y_test = import_data(path_to_data, model='CNN')
    cnn = CNN_Model(x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test,
                    rul_pre=rul_pre,
                    discrete_candidate=discrete_candidate,
                    optimization=True,
                    epoch=epochs)
    sa = SA(cnn.cnn_get_error, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            rul_pre=rul_pre, output_image=output_image,
            initial_temp=initial_temp,
            final_temp=final_temp, alpha=alpha, max_iter=max_iter, var_size=var_size,
            candidate=discrete_candidate, net="CNN", save_path=save_path)
    # objective, initial_temp, final_temp, alpha, max_iter, var_size, candidate, net
    params_dict = sa.run()
    return cnn, params_dict


def run_cnn_ga(x_train, x_test, y_train, y_test, discrete_candidate, var_size,
               rul_pre=False, output_image=0, epochs=5,
               threshold=100.9960, dna_size=9, pop_size=6,
               cross_rate=0.3, mutation_rate=0.1, n_generations=10,
               save_path="./"):
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
    # x_train, x_test, y_train, y_test = import_data(path_to_data, model='CNN')

    # 2.模型准备
    cnn = CNN_Model(x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test,
                    rul_pre=rul_pre,
                    discrete_candidate=discrete_candidate,
                    optimization=True,
                    epoch=epochs)

    # 4.开始优化
    from moduls.ml.ga import run
    best_generation, dna_set = run(cnn, Model_name="CNN", var_size=var_size,
                                   rul_pre=rul_pre, output_image=output_image,
                                   threshold=threshold, dna_size=dna_size, pop_size=pop_size,
                                   cross_rate=cross_rate, mutation_rate=mutation_rate,
                                   n_generations=n_generations, save_path=save_path, )

    # 5.优化结果参数提取

    param_dict = {"dropout": dna_set[0][0],
                  "learning_rate": dna_set[0][1],
                  "batch_size": dna_set[0][2],
                  "number_of_convolution": dna_set[0][3],
                  "epochs": random.randint(5,10)
                  }
    return cnn, param_dict


def CNN(x_train, x_test, y_train, y_test, opt_option=None, rul_pre=False,
        output_image=0,
        dropout=0.5, learning_rate=0.002,
        batch_size=32, conv=6, epochs=5,
        pso_part_num=2, pso_num_itr=5, sa_initial_temp=500, sa_final_temp=1,
        sa_alpha=0.9, sa_max_iter=5,
        ga_threshold=100.9960, ga_dna_size=9, ga_pop_size=6, ga_cross_rate=0.3,
        ga_mutation_rate=0.1, ga_n_generations=1,
        save_path=None):
    """
    Main function to call the selected model and optimizer

    :param path_to_data: list, upper and under boundaries of all variables
    :param opt_option: string, 'PSO' or 'SA' 优化方法
    :param gamma: 惩罚系数
    :param C: 核函数系数
    :param save_path: string, 图片保存路径
    :return: None
    """
    # 参数校验
    if not opt_option:
        if dropout <= 0 or dropout >= 1:
            abort(400, "ERROR: The value range of parameter dropout is (0, 1).")
        if learning_rate <= 0 or learning_rate >= 1:
            abort(400, "ERROR: The value range of parameter learning rate is (0, 1).")
        # if epochs < 2:
        #     abort(400, "The parameter epochs must be a positive integer and greater than 2.")
        # if batch_size < 2:
        #     abort(400, "The parameter batch size must be a positive integer and greater than 2.")

    # below should get from config
    # dropout = params[0]
    # learning_rate = params[1]
    # batch_size = params[2]
    # conv = params[3]
    do = [0.3, 0.8]  # dropout
    lr = [0.0001, 0.02]  # learning rate
    # 6 candidate of batch size
    # bs_candidate = [1, 16, 32, 64, 128, 256]  # batch size
    bs_candidate = list(range(1,256))
    print("bs_candidate", bs_candidate)
    # conv_candidate = [4, 6, 8]  # convolution 卷积
    conv_candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    discrete_candidate = [bs_candidate, conv_candidate]
    # 3 candidate of number of the convolution layers
    var_size = [do, lr, [1, 256], [0, 1]]  # [dropout, learning rate, batch size, conv]

    ########## x_train, x_test, y_train, y_test = import_data(path_to_data, model='CNN')
    ###############
    # size_train = x_train.shape[0]
    # size_test = x_test.shape[0]
    # x_train = x_train.reshape(size_train, x_train.shape[1], 1)
    # x_test = x_test.reshape(size_test, x_test.shape[1], 1)
    #
    # y_train_max = int(max(y_train)) + 1
    # y_test_max = int(max(y_test)) + 1
    # from moduls.ml.dataset import to_cat
    # y_train = to_cat(y_train, num_classes=y_train_max)
    # y_test = to_cat(y_test, num_classes=y_test_max)

    # print("x_train:", x_train.shape)
    # print("y_train", y_train)
    # print("y_test", y_test)
    ###########################
    if opt_option == 'PSO':

        cnn, params_dict = run_cnn_pso(x_train, x_test, y_train, y_test, var_size,
                                       discrete_candidate, rul_pre=rul_pre, epochs=epochs,
                                       output_image=output_image,
                                       part_num=pso_part_num,
                                       num_itr=pso_num_itr, save_path=save_path)
        # 使用优化后得参数训练模型，打印结果
        params = [params_dict["dropout"],
                  params_dict["learning_rate"],
                  params_dict["batch_size"],
                  params_dict["number_of_convolution"]]

        y, error = cnn.get_score(params)
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
        cnn, params_dict = run_cnn_sa(x_train, x_test, y_train, y_test, var_size,
                                      discrete_candidate, rul_pre=rul_pre, epochs=epochs,
                                      output_image=output_image,
                                      initial_temp=sa_initial_temp,
                                      final_temp=sa_final_temp, alpha=sa_alpha,
                                      max_iter=sa_max_iter, save_path=save_path)
        # 使用优化后得参数训练模型，打印结果

        params = [params_dict["dropout"],
                  params_dict["learning_rate"],
                  params_dict["batch_size"],
                  params_dict["number_of_convolution"]]

        y, error = cnn.get_score(params)
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
        cnn, params_dict = run_cnn_ga(x_train, x_test, y_train, y_test,
                                      discrete_candidate, rul_pre=rul_pre, epochs=epochs,
                                      output_image=output_image,
                                      var_size=var_size, threshold=ga_threshold, dna_size=ga_dna_size,
                                      pop_size=ga_pop_size, cross_rate=ga_cross_rate,
                                      mutation_rate=ga_mutation_rate,
                                      n_generations=ga_n_generations,
                                      save_path=save_path)
        # 使用优化后得参数训练模型，打印结果

        params = [params_dict["dropout"],
                  params_dict["learning_rate"],
                  params_dict["batch_size"],
                  params_dict["number_of_convolution"]]

        y, error = cnn.get_score(params)
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

        cnn = CNN_Model(x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test,
                        rul_pre=rul_pre, epoch=epochs,
                        discrete_candidate=discrete_candidate,
                        # optimization=False
                        )
        params = [dropout, learning_rate, batch_size, conv]
        y, error = cnn.get_score(params)
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

        params_dict = {"dropout": params[0], "learning_rate": params[1],
                       "batch_size": params[2], "number_of_convolution": params[3],
                       "epochs": epochs}
        params_dict["accuracy"] = acc
        return y_pre, params_dict

    else:
        print("without this optimization option")
        raise


if __name__ == '__main__':
    """
    Main function to call the selected model and optimizer
    """
    # Main function
    opt_option = None  # choose between "PSO" and "SA" "GA"
    # data_path = './Dataset/'
    # save_path = './test/'
    #
    # x_train, x_test, y_train, y_test = import_data(data_path, model='CNN')

    # from test import read_data_csv
    # data_path = "./Dataset/feature_1_1_all.csv"
    # x_train, x_test, y_train, y_test = read_data_csv(data_path)
    # size_train = x_train.shape[0]
    # size_test = x_test.shape[0]
    # x_train = x_train.reshape(size_train, x_train.shape[1], 1)
    # x_test = x_test.reshape(size_test, x_test.shape[1], 1)
    # y_train = y_train.reshape(size_train, 1)
    # y_test = y_test.reshape(size_test, 1)

    use_data = read_time_domain_signal("/home/python/Desktop/PHM_Dev/test_data/fault_diagnose/testdata.mat")
    use_data_label = read_time_domain_signal("/home/python/Desktop/PHM_Dev/test_data/fault_diagnose/testlabel.mat")
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(use_data,
                                                        use_data_label,
                                                        test_size=0.25,
                                                        random_state=0)

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

    """
        rul_pre:
            input:
            x(2327,38,1)
            y(2327,1)
    """

    res = CNN(x_train, x_test, y_train, y_test, rul_pre=False, output_image=0,
              opt_option=opt_option, save_path=None, sa_max_iter=5)
    print("最终分数为：", res)

    """
    # below should get from config
    do = [0.3, 0.8]  # dropout
    lr = [0.0001, 0.02]  # learning rate
    # 6 candidate of batch size
    conv_candidate = [4, 6, 8]  # convolution 卷积
    bs_candidate = [1, 16, 32, 64, 128, 256]  # batch size
    discrete_candi = [bs_candidate, conv_candidate]
    # 3 candidate of number of the convolution layers
    var_size = [do, lr, [0, 1], [0, 1]]  # lower and upper bounds of all params
    """
