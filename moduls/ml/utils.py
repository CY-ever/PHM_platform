import itertools
import os
import random

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import numpy.fft as nf
from scipy.signal import find_peaks
from sklearn.metrics import classification_report


def plot_learning_curve(history):
    """
    Plot the learning curve.

    :param history: Keras API model training history
    :return: None
    """
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


def to_cat(data, num_classes=None):
    """
    Change the label to One-hot coding.

    :param data: label to change as in [1, 2, 3, 4]
    :param num_classes: total numbers of classes
    :return: Encoded label
    """
    # Each data should be represents the class number of the data
    if num_classes is None:
        num_classes = np.unique(data)
    data_class = np.zeros((data.shape[0], num_classes))
    for i in range(data_class.shape[0]):
        num = data[i]
        data_class[i, num] = 1
    return data_class


def report(self, data, labels):
    """
    print out test report

    :param data: test data
    :param labels: test label
    """
    print(
        classification_report(np.argmax(labels, axis=1),
                              np.argmax(self.model.predict(data), axis=1),
                              digits=4))


def translate_params(params, candidate):
    """
    Translate the list of parameters to the corresponding parameter.

    :param candidate: list, discrete choices of one parameter in network
    :param params: list, [dropout, learning_rate, batch_size, number of convolution]
    :return: value of dropout(float), learning_rate(float) and batch_size(int)
    """
    conv_candidate = candidate[1]
    bs_candidate = candidate[0]
    dropout = params[0]
    learning_rate = params[1]
    conv, batch_size = None, None
    for j in range(len(bs_candidate)):
        if (j / len(bs_candidate)) <= params[2] <= ((j + 1) / len(bs_candidate)):
            batch_size = bs_candidate[j]
        elif params[2] == 1:
            batch_size = bs_candidate[-1]
        elif params[2] > 1:
            batch_size = params[2]
    for i in range(len(conv_candidate)):
        if (i / len(conv_candidate)) <= params[3] <= ((i + 1) / len(conv_candidate)):
            conv = conv_candidate[i]
        elif params[3] == 1:
            conv = conv_candidate[-1]
        elif params[3] > 1:
            conv = params[3]
    assert conv in conv_candidate
    print("batch_size", batch_size)
    assert batch_size in bs_candidate

    return dropout, learning_rate, batch_size, conv


def print_params(params, candidate, net=None):
    """
    print the network's params via translating function

    :param candidate: list, discrete choices of one parameter in network
    :param net: String, choose between "CNN", "DBN", "SVM"
    :param params: list of network's parameters
    :return: None
    """
    if net == "CNN":
        # dropout, learning_rate, batch_size, conv = translate_params(params, candidate)
        dropout, learning_rate, batch_size, conv = params[0], params[1], params[2], params[3]
        epochs = random.randint(5,20)
        print("Best parameters: ",
              "\ndropout=", dropout,
              "learning rate=", learning_rate,
              "batch size=", batch_size,
              "number of convolution=", conv,
              "epochs", epochs)

        return {"dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "number_of_convolution": conv,
                "epochs": epochs}

    elif net == "DBN":
        print('Best parameters: ',
              '\nDropout=', params[0],
              'LearningRate_RBM=', params[1],
              'LearningRate_nn=', params[2])

        return {"Dropout": params[0],
                "LearningRate_RBM": params[1],
                "LearningRate_nn": params[2]}

    elif net == "SVM":
        print("Best Parameters: ",
              "\nC=",  params[0],
              "gamma=",  params[1])

        return {"C": params[0],
                "gamma": params[1]}

    elif net == "KNN":
        print("Best Parameters: ",
              "\nK=",  params[0],
              "weights=",  params[1])

        return {"K": params[0],
                "weights": params[1]}

    elif net == "AE":
        print("Best Parameters: ",
              "\nLayerCount=", params[0],
              "\nunits1=", params[1],
              "\nunits2=", params[2],
              "\nunits3=", params[3],
              "\nepochs=", params[4],
              "\nbatchSize=", params[5],
              "\ndenseActivation=", params[6],
              "\noptimizer=", params[7],
              "\nloss=", params[8])

        return {"LayerCount": params[0],
                "units1": params[1],
                "units2": params[2],
                "units3": params[3],
                "epochs": params[4],
                "batchSize": params[5],
                "denseActivation": params[6],
                "optimizer": params[7],
                "loss": params[8],
                }

    elif net in ["RF", "ET"]:
        print("Best Parameters: ",
              "\nmax_depth=", params[0],
              "max_leaf_nodes=", params[1],
              "n_estimators=", params[2])

        return {"max_depth": params[0],
                "max_leaf_nodes": params[1],
                "n_estimators": params[2]}

    elif net == "DT":
        print("Best Parameters: ",
              "\nmax_depth=", params[0],
              "max_leaf_nodes=", params[1],)

        return {"max_depth": params[0],
                "max_leaf_nodes": params[1],}

    elif net == "LSTM":
        print("Best Parameters: ",
              "\nlstm_count:", params[0],
              "\nunits1:", params[1],
              "\nunits2:", params[2],
              "\nunits3:", params[3],
              "\ndropoutRate:", params[4],
              "\nepochs:", params[5],
              "\nbatchSize:", params[6],
              "\ndenseActivation:", params[7],
              "\noptimizer:", params[8],
              "\nloss:", params[9])

        return {"lstm_count": params[0],
                "units1": params[1],
                "units2": params[2],
                "units3": params[3],
                "dropoutRate": params[4],
                "epochs": params[5],
                "batchSize": params[6],
                "denseActivation": params[7],
                "optimizer": params[8],
                "loss": params[9],
                }


def data_FFT(data):
    """
    use fourier transformation to change dataset from time domain into frequency domain

    :param data: input original data
    :return: transformed data
    """
    data_fft = []
    for i in range(len(data)):
        rank_i = data[i]
        # print(rank1)
        times = np.arange(rank_i.size)
        freqs = nf.fftfreq(times.size, times[1] - times[0])
        xs = np.abs(freqs)
        complex_array = nf.fft(rank_i)
        ys = np.abs(complex_array)
        # ## plot signal in time domain
        # plt.figure()
        # plt.plot(times, rank_i)
        # plt.title("Signal[0] in Time Domain")
        # plt.xlabel("Time")
        # plt.ylabel("Amplitude")
        # plt.show()
        # ## plot signal in frequency domain
        # plt.figure()
        # plt.plot(xs, ys)
        # plt.xlabel("Frequency")
        # plt.title('Frequency Domain', fontsize=16)
        # plt.ylabel('Amplitude', fontsize=12)
        # plt.tick_params(labelsize=10)
        # plt.grid(linestyle=':')
        # plt.show()

        ## find peaks in frequency domain
        peak_id, peak_property = find_peaks(ys, height=6, distance=10)
        peak_freq = xs[peak_id]
        peak_height = peak_property['peak_heights']
        peak_freq = np.unique(peak_freq)
        if peak_freq is not None:
            peak_freq = np.append(peak_freq[0], peak_freq[-1])  # select first and last peaks
        peak_height = np.unique(peak_height)
        if peak_height is not None:
            peak_height = np.append(peak_height[0], peak_height[-1])
        else:
            print("peak_freq not found, change params")
        # print('peak_freq',peak_freq)
        # print('peak_height',peak_height)
        data_i_fft = np.append(peak_freq, peak_height)
        # print(data_i_fft)
        data_fft.append(data_i_fft)
    data_fft = np.asarray(data_fft).reshape(len(data), 4)  # generate new x_train from frequency domain
    # print(data_fft)
    return data_fft


# 绘制混淆矩阵
def plot_confuse(y_pre, y_true, output_image=0, save_path=None):
    """
    绘制混淆矩阵
    :param y_pre: 预测值
    :param y_val: 真实值
    :return:
    """
    from sklearn.metrics import confusion_matrix
    #获得真实标签

    # truelabel = y_true.argmax(axis=-1) # 将one-hot转化为label
    if y_pre.shape == y_true.shape:
        truelabel = y_true
    else:
        truelabel = y_true.T[0]
    predictions = y_pre
    # predictions = y_pre.argmax(axis=-1)

    print("y_true:", truelabel)
    print("y_pred:", predictions)
    print("y_true:", set(list(truelabel)))
    print("y_pred:", set(list(predictions)))
    cm = confusion_matrix(y_true=truelabel, y_pred=predictions)
    # y_true: 是样本真实分类结果，y_pred: 是样本预测分类结果
    # y_pred：预测结果
    # labels：是所给出的类别，通过这个可对类别进行选择
    # sample_weight : 样本权重
    print(cm, type(cm))

    plt.figure()

    # 指定分类类别
    # print("truelabel:", truelabel.shape)
    # print("truelabel:", truelabel)
    # classes = range(np.max(truelabel)+1)
    classes = list(set(list(truelabel)))
    title='Confusion matrix'

    #混淆矩阵颜色风格

    # cmap = plt.jet()
    cmap = plt.viridis()
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontdict={'weight': "normal", 'size': 15})

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.

    # 按照行和列填写百分比数据

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, '{:.2f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    #left，right，up，bottom
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.ylabel('True label', fontdict={'weight': "normal", 'size': 12})

    plt.xlabel('Predicted label', fontdict={'weight': "normal", 'size': 12})

    # if save_path:
    #     # 判断图片保存类型
    #     if output_image == 1:
    #         plt.savefig(os.path.join(save_path, "confusion_matrix.jpg"))
    #     elif output_image == 2:
    #         plt.savefig(os.path.join(save_path, "confusion_matrix.svg"))
    #     elif output_image == 3:
    #         plt.savefig(os.path.join(save_path, "confusion_matrix.pdf"))
    #     else:
    #         plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    #
    # else:
    #     plt.show()

    if output_image == 0:
        file_name = f"confusion_matrix.png"
        path = os.path.join(save_path, file_name)
        plt.savefig(path)
    elif output_image == 1:
        file_name1 = f"confusion_matrix.png"
        file_name2 = f"confusion_matrix.jpg"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 2:
        file_name1 = f"confusion_matrix.png"
        file_name2 = f"confusion_matrix.svg"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 3:
        file_name1 = f"confusion_matrix.png"
        file_name2 = f"confusion_matrix.pdf"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    plt.close()

    # plt.close()


# 绘制RUL预测曲线
def plot_rul(y_pre, y_true, output_image=0, save_path=None):

    # 计算最大误差,平均误差
    y_true = y_true.reshape((-1,))
    y_pre = y_pre.reshape((-1,))
    print("y_pre", y_pre.shape)
    print("y_true", y_true.shape)
    error_list = np.abs(y_pre - y_true)
    mean_error = np.sum(error_list)/error_list.shape[0]
    max_error = error_list.max()
    print("回归模型预测的最大误差是:", max_error)
    print("回归模型预测的平均误差是:", mean_error)
    # print("回归模型预测的全部误差是:", error_list)

    x_sticks = np.linspace(0, 1, len(y_pre))

    plt.figure()

    # print("y_true:", y_true.shape)
    # print("y_pre:", y_pre.shape)
    y_true = list(y_true)
    y_pre = list(y_pre)
    # 合并数组
    df = pd.DataFrame({'y_true': y_true, 'y_pre': y_pre})
    #
    # print("df", df)
    # 按照其中一列降序排序
    df.sort_values(by='y_true', ascending=False, inplace=True)
    y_true = np.array(df['y_true'])

    # df.sort_values(by='y_pre', ascending=False, inplace=True)

    y_pre = np.array(df['y_pre'])
    # print("y_true", y_true)
    # print("y_pre", y_pre)

    # y_pre = abs(np.sort(-y_pre, axis=0))

    # 绘制折线图
    plt.plot(y_true*100, label="True value")
    plt.plot(y_pre*100,  label="Predicted value")

    plt.ylabel("RUL [%]")
    plt.xlabel("Sample points")
    plt.title("RUL prediction curve")

    plt.legend()

    if output_image == 0:
        file_name = f"rul_figure.png"
        path = os.path.join(save_path, file_name)
        plt.savefig(path)
    elif output_image == 1:
        file_name1 = f"rul_figure.png"
        file_name2 = f"rul_figure.jpg"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 2:
        file_name1 = f"rul_figure.png"
        file_name2 = f"rul_figure.svg"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    elif output_image == 3:
        file_name1 = f"rul_figure.png"
        file_name2 = f"rul_figure.pdf"
        path1 = os.path.join(save_path, file_name1)
        path2 = os.path.join(save_path, file_name2)
        plt.savefig(path1)
        plt.savefig(path2)
    plt.close()

