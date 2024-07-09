from sklearn.metrics import r2_score
from sklearn.svm import SVC, SVR

from sklearn.model_selection import train_test_split


class SVM_Model:
    def __init__(self, x_train, y_train, x_test, y_test, rul_pre=False, optimization=True):
        """

        :param x_train: training set
        :param y_train: training label
        :param x_test: test set
        :param y_test: test label
        :param optimization: bool
        """
        self.x_train = x_train
        self.y_train = y_train.ravel()  # 将label转为一维数组, shape: (11821, 1)-->(11821,)
        self.rul_pre = rul_pre

        if x_test is None:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_train, self.y_train, random_state=42, test_size=0.2)
        else:
            self.x_test = x_test
            self.y_test = y_test.ravel()
        self.model = None
        # self.optimization = optimization

    def get_score(self, params):
        """
        Used as objective function, get model score

        :param params: list, Model hyperparameters
        :return: float, model error
        """
        # assert self.optimization is True
        # kernel = {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
        if not self.rul_pre:
            self.model = SVC(C=params[0], gamma=params[1])
        else:
            self.model = SVR(C=params[0], gamma=params[1])
        self.model.fit(self.x_train, self.y_train)  # 训练模型
        y_pre = self.model.predict(self.x_test)  # 对测试集进行分类预测
        if not self.rul_pre:
            score = self.model.score(self.x_test, self.y_test)
        else:
            score = r2_score(self.y_test, y_pre)

        Error = 1 - score  # 计算测试分类正确率
        return Error

    def train_svm(self, params):
        """
        Usually train SVM

        :param params: list, model params
        :return: None
        """
        # assert self.optimization is False
        if not self.rul_pre:
            self.model = SVC(C=params[0], gamma=params[1])
        else:
            self.model = SVR(C=params[0], gamma=params[1])
        self.model.fit(self.x_train, self.y_train)  # 训练模型
        # 预测结果
        y_pre = self.model.predict(self.x_test) # 对测试集进行分类预测
        # 预测正确率
        if not self.rul_pre:
            score = self.model.score(self.x_test, self.y_test)
        else:
            score = r2_score(self.y_test, y_pre)
        print("Training complete, with accuracy:", score)
        return y_pre, score
