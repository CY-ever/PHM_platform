import numpy as np
import scipy.io as scio
# from utils import to_cat
from sklearn.preprocessing import StandardScaler


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
        num = int(data[i])
        # print("num:", num)
        # print("data_class.shape:", data_class.shape)
        # print("data:", data)
        data_class[i, num] = 1
    return data_class


def import_data(path, model=None):
    """
    Import data from given path.
    :param path: string, Folder of the data files
    :param model: which model do we use
    :return: ndarray, training and test data
    """
    # import dataset
    x_train_file = path + 'traindata.mat'
    x_train_dict = scio.loadmat(x_train_file)
    x_train_all = x_train_dict.get('train_data')

    y_train_file = path + 'trainlabel.mat'
    y_train_dict = scio.loadmat(y_train_file)
    y_train_all = y_train_dict.get('train_label')

    x_test_file = path + 'testdata.mat'
    x_test_dict = scio.loadmat(x_test_file)
    x_test_all = x_test_dict.get('test_data')

    y_test_file = path + 'testlabel.mat'
    y_test_dict = scio.loadmat(y_test_file)
    y_test_all = y_test_dict.get('test_label')
    # loading complete
    # how many classes
    classes = np.unique(y_train_all)
    # x_train_all, x_vali, y_train_all, y_vali = train_test_split(x_train_all, y_train_all, test_size=0.9)
    if model in ['CNN', ]:
        size_train = x_train_all.shape[0]
        size_test = x_test_all.shape[0]
        x_train = x_train_all.reshape(size_train, 576, 1)
        x_test = x_test_all.reshape(size_test, 576, 1)
        y_train = to_cat(y_train_all, num_classes=4)
        y_test = to_cat(y_test_all, num_classes=4)
        return x_train, x_test, y_train, y_test

    if model in ['LSTM', ]:
        size_train = x_train_all.shape[0]
        size_test = x_test_all.shape[0]
        x_train = x_train_all.reshape(size_train, 1, 576)
        x_test = x_test_all.reshape(size_test, 1, 576)
        # x_train = x_train_all.reshape(size_train, 576)
        # x_test = x_test_all.reshape(size_test, 576)
        y_train = to_cat(y_train_all, num_classes=4)
        y_test = to_cat(y_test_all, num_classes=4)
        # y_train = y_train.astype("int")
        # y_test = y_test.astype("int")
        # y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1])
        # y_test = y_test.reshape(y_test.shape[0], 1, y_test.shape[1])

        return x_train, x_test, y_train, y_test

    elif model in ['DBN', 'AE']:
        y_train = to_cat(y_train_all, num_classes=4)
        y_test = to_cat(y_test_all, num_classes=4)
        return x_train_all, x_test_all, y_train, y_test

    elif model in ['SVM', 'KNN', 'RF', "DT"]:
        # 特征之标准化
        transfer = StandardScaler()
        x_train_all = transfer.fit_transform(x_train_all)
        x_test_all = transfer.fit_transform(x_test_all)

        return x_train_all, x_test_all, y_train_all, y_test_all

    else:
        return x_train_all, x_test_all, y_train_all, y_test_all
