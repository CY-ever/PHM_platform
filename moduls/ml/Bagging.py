import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, AdaBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from moduls.ml.dataset import import_data
from moduls.ml.utils import data_FFT
import pickle
import os


# def run_lr(path_to_data):
#
#     """
#     Main function KNN_Model.
#
#     :param path_to_data: string, Folder of the data files
#     :return: None
#     """
#     x_train, x_test, y_train, y_test = import_data(path_to_data, model='KNN')
#     x_train = data_FFT(x_train)  # fft on original dataset
#     x_test = data_FFT(x_test)  # fft on original dataset
#     knn = RIDGE_Model(x_train, y_train, x_test, y_test)
#     k_range = range(1, 25)  # user input (under, upper bounds)
#     weight_choices = ["uniform", "distance"]  # user input, string in list
#     params = [k_range, weight_choices]  # editable params
#     # pre = knn.predict(params)  # user can decide to plot/save result or not
#     pre, Error = knn.get_score(params)
#     print("准确率为：", 1-Error)
#     plot = False    # save: bool, choose to save or not
#     save = False    # plot: bool, choose to plot or not
#     load = False    # load: bool, choose to load or not
#     if plot:
#         knn.plot_curve()
#     if save:
#         knn.save_result()
#     if load:
#         knn.load_model()


class Bagging_Model:

    def __init__(self, x_train, y_train, x_test, y_test, outdir=None):
        """
        :param x_train: training data
        :param y_train: training label
        :param x_test: test data
        :param y_test: test label
        :param outdir: dir of output
        """
        # print('Selected Network : KNN_Model')
        self.x_train = x_train
        self.y_train = y_train.ravel()
        self.x_test = x_test
        self.y_test = y_test.ravel()
        self.result_error = 0  # initialisation
        self.result_k = 0  # initialisation
        self.k_error = []
        self.k_err_train = []
        self.k_range = 1
        # self.path = path
        self.et = None
        self.filename = 'kridge_model.sav'
        # if os.path.exists(outdir):
        #     self.outdir = outdir
        # else:
        #     os.makedirs(outdir)
        #     self.outdir = outdir


    def plot_curve(self):
        """
        plot Error_list on variable k value

        :return: None
        """
        plt.plot(self.k_range, self.k_error)
        plt.title("Error under different choice of K")
        plt.xlabel("Value of K for KNN_Model")
        plt.ylabel("Error")
        plt.show()

    def save_result(self):
        """
        save result of knn into path
        :param path: string, path of dataset
        """
        pickle.dump(self.et, open(self.outdir + self.filename, 'wb'))

    def load_model(self):
        """
        :return:
        """
        try:
            loaded_model = pickle.load(open(self.filename, 'rb'))
            result = loaded_model.score(self.x_test, self.y_test)
            error = 1 - result
            return error
        except:
            print("No such file or directory.")

    def get_score(self, params, rul_pre=False):
        """
        Used as objective function, get model score

        :param params: list, Model hyperparameters
        :return: float, model error
        """
        # assert self.optimization is True
        # n_estimators=5
        if not rul_pre:
            self.et = BaggingClassifier(n_estimators=10
                                        )
            self.et.fit(self.x_train, self.y_train)  # 训练模型
            pre = self.et.predict(self.x_test)  # 对测试集进行分类预测
            Error = 1 - self.et.score(self.x_test, self.y_test)  # 计算测试分类正确率
        else:
            self.et = BaggingRegressor()
            self.et.fit(self.x_train, self.y_train)  # 训练模型
            pre = self.et.predict(self.x_test)  # 对测试集进行分类预测
            score = r2_score(self.y_test, pre)
            Error = 1 - score

        return pre, Error


if __name__ == "__main__":
    """
    Main function to call the knn
    """
    path = './dataset/'
    # run_lr(path)
    # get_score()
