import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LinearRegression, SGDRegressor, LogisticRegression
from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier,  SGDClassifier, MultiTaskElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from moduls.ml.dataset import import_data
from moduls.ml.utils import data_FFT
import pickle
import os


def run_lr(path_to_data):

    """
    Main function KNN_Model.

    :param path_to_data: string, Folder of the data files
    :return: None
    """
    x_train, x_test, y_train, y_test = import_data(path_to_data, model='KNN')
    # x_train = data_FFT(x_train)  # fft on original dataset
    # x_test = data_FFT(x_test)  # fft on original dataset
    knn = KNN_Model(x_train, y_train, x_test, y_test)
    k_range = range(1, 25)  # user input (under, upper bounds)
    weight_choices = ["uniform", "distance"]  # user input, string in list
    params = [k_range, weight_choices]  # editable params
    # pre = knn.predict(params)  # user can decide to plot/save result or not
    pre, Error = knn.get_score(params)
    print("准确率为：", 1-Error)
    plot = False    # save: bool, choose to save or not
    save = False    # plot: bool, choose to plot or not
    load = False    # load: bool, choose to load or not
    if plot:
        knn.plot_curve()
    if save:
        knn.save_result()
    if load:
        knn.load_model()


class KNN_Model:

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
        self.lr = None
        self.filename = 'knn_model.sav'
        # if os.path.exists(outdir):
        #     self.outdir = outdir
        # else:
        #     os.makedirs(outdir)
        #     self.outdir = outdir

    def predict(self, params):
        """

        :param params: params[0] is number of neighbors;
                    params[1] is weight used in prediction,
                        - "uniform" all points have the same weight,
                        - "distance" weights by the inverse of distance,
                        - [callable] : a user-defined function which accepts an array of distances;
        :return: Error, k value
        """
        # get params
        k_range, weight_choices = params
        self.k_range = k_range
        # Feature Scaling
        # sc = StandardScaler()
        # self.x_train = sc.fit_transform(self.x_train)
        # self.x_test = sc.transform(self.x_test)

        # Fitting K-NN to the Training set

        # lr = Ridge()
        # lr = ElasticNet()
        lr = Lasso()
        # params[2] and params[3] Default is "minkowski" with p=2, equivalent to the standard Euclidean metric
        # e.g. [5, "uniform",'minkowski', 2]
        # only the first and second params are set to be changed by user
        self.lr = lr
        lr.fit(self.x_train, self.y_train)  # train on train_file
        err = 1 - lr.score(self.x_train, self.y_train)
        Error = 1 - lr.score(self.x_test, self.y_test)  # error on test_file
        # print("k=", k)
        # print("weights=", weights)
        print("Error", Error)
        self.k_error.append(Error)
        self.k_err_train.append(err)
        self.result_error = min(self.k_error)
        self.result_k = self.k_error.index(self.result_error)+1
        print("best result", "k=", self.result_k, "lowest error=", self.result_error)

        return self.result_error, self.result_k

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
        pickle.dump(self.knn, open(self.outdir+self.filename, 'wb'))

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

    def get_score(self, params):
        """
        Used as objective function, get model score

        :param params: list, Model hyperparameters
        :return: float, model error
        """
        # assert self.optimization is True
        # kernel = {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
        k = params[0]
        weights = params[1]

        alphas_to_test = np.linspace(0.01, 1e9, num=50)  # 现在我们有50个test用的点
        # {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'},
        # self.lr = RidgeClassifier(solver="saga",
        #                           tol=1e-3,
        #                           alpha=10000.0)
        # self.lr = RidgeClassifierCV()
        self.lr = RidgeClassifierCV(alphas=alphas_to_test, store_cv_values=True)

        # self.lr = KMeans(n_clusters=4)
        # self.lr = MultiTaskElasticNet()
        # self.lr = Lasso()
        # self.lr = LinearRegression()
        # loss: 'hinge'0.45, 'log'0.43, 'modified_huber'0.4,
        #         'squared_hinge'0.405, 'perceptron'0.409, or a regression loss: 'squared_loss',
        #         'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'
        # self.lr = SGDClassifier(loss='squared_loss',  # default='hinge'
        #                         fit_intercept=True,
        #                         learning_rate='invscaling',
        #                         eta0=0.1,
        #                         max_iter=1000)
        # self.lr = SGDClassifier(loss='huber',
        #                         fit_intercept=True,
        #                         learning_rate='invscaling',
        #                         eta0=0.1)
        # 'newton-cg'0.605, 'sag', 'saga' and 'lbfgs'0.597 'liblinear'0.589
        # self.lr = LogisticRegression(solver='newton-cg', penalty='l2', C=1e4, max_iter=1e5)
        # self.lr = Ridge()
        i = 0
        sc = []
        while i < 1:
            i += 1
            self.lr.fit(self.x_train, self.y_train)  # 训练模型
            pre = self.lr.predict(self.x_test) # 对测试集进行分类预测
            # res = calinski_harabasz_score(self.x_test, self.y_test)
            Error = 1 - self.lr.score(self.x_test, self.y_test)  # 计算测试分类正确率
            print("准确率为：", 1-Error)
            x = 1 - Error
            sc.append(x)
        sc = np.array(sc)
        print("平均正确率为", sc.mean())
        return pre, Error


if __name__ == "__main__":
    """
    Main function to call the knn
    """
    path = './dataset/'
    run_lr(path)
    # get_score()
