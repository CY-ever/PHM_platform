import os
import shutil  # 更新缓存地址
import pickle  # 保存模型

import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neural_network import BernoulliRBM
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from moduls.ml.utils import plot_learning_curve


class DBN_Model:

    def __init__(self,
                 train_data,
                 targets,
                 rbm_layers,
                 batch_size,
                 rul_pre=False,
                 x_test=None,
                 y_test=None,
                 outdir=None,
                 logdir=None,
                 optimization=True):
        """

        :param train_data,targets:训练集数据
        :param rbm_layers:list 指定rbm层数及每层节点数目，[400,200,50]
        :param batch_size: int
        :param outdir: 输出目录
        :param logdir: 缓存目录
        :param optimization: bool Ture:优化算法；False：直接调用DBN
        """
        self.data = train_data
        self.x_test = x_test
        self.y_test = y_test
        self.hidden_sizes = rbm_layers
        self.targets = targets
        self.rul_pre = rul_pre
        self.outputs = targets.shape[1]
        # self.outputs = targets.shape[0]

        self.batch_size = batch_size

        self.rbm_weights = []
        self.rbm_biases = []
        self.rbm_h_act = []

        self.model = None
        self.history = None
        self.Error = None
        self.optimization = optimization
        self.val_acc = []

        if self.optimization is False:
            if os.path.exists(outdir):
                shutil.rmtree(outdir)
            if os.path.exists(logdir):
                shutil.rmtree(logdir)
            os.makedirs(outdir)
            os.makedirs(logdir)
        #
        self.outdir = outdir
        self.logdir = logdir

    def pretrain(self, params, verbose):  # 预训练,构造n层RBM网络
        """
        Pretraining, build n-layers RBM network

        :param params:params[1] rbm 学习率
        :param verbose: 日志显示
        """
        visual_layer = self.data

        for i in range(len(self.hidden_sizes)):

            if self.optimization is False:
                print("[RBM] Layer {} Pre-Training".format(i + 1))
                print('visual_layer: ', visual_layer.shape)

            rbm = BernoulliRBM(n_components=self.hidden_sizes[i],
                               n_iter=10,
                               learning_rate=params[1],
                               verbose=verbose,
                               batch_size=self.batch_size)
            rbm.fit(visual_layer)

            self.rbm_weights.append(rbm.components_)
            self.rbm_biases.append(rbm.intercept_hidden_)
            self.rbm_h_act.append(rbm.transform(visual_layer))
            visual_layer = self.rbm_h_act[-1]

            if self.optimization is False:
                print('hidden_layer: ', visual_layer.shape)
                with open(self.outdir + "rbm_weights.p", 'wb') as f:
                    pickle.dump(self.rbm_weights, f)

                with open(self.outdir + "rbm_biases.p", 'wb') as f:
                    pickle.dump(self.rbm_biases, f)

                with open(self.outdir + "rbm_hidden.p", 'wb') as f:
                    pickle.dump(self.rbm_h_act, f)

    def finetune(self, params, verbose, rul_pre=False):  # 微调
        """
        Finetuning RBM network

        :param params: params[0]：Dropout；params[2]：nn学习率
        :param verbose: bool, show training progress
        :return: None
        """
        model = Sequential()
        for i in range(len(self.hidden_sizes)):
            if i == 0:
                model.add(Dense(self.hidden_sizes[i],
                                # TODO
                                # kernel_regularizer=regularizers.l2(0.001),  # 正则化
                                activation='relu',
                                input_dim=self.data.shape[1],
                                name='rbm_{}'.format(i)))

                model.add(BatchNormalization())
            else:
                model.add(
                    Dense(self.hidden_sizes[i],
                          # TODO
                          # kernel_regularizer=regularizers.l2(0.001),
                          activation='relu',
                          name='rbm_{}'.format(i)))
                model.add(BatchNormalization())
        model.add(keras.layers.AlphaDropout(rate=params[0]))  # Dropout

        if not self.rul_pre:
            model.add(Dense(self.outputs, activation='softmax'))
            opt = keras.optimizers.Adam(learning_rate=params[2])
            model.compile(optimizer=opt,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        else:
            model.add(Dense(self.outputs, activation='relu'))
            opt = keras.optimizers.Adam(learning_rate=params[2])
            model.compile(optimizer=opt,
                          loss='mse',
                          metrics=['accuracy'])

        for i in range(len(self.hidden_sizes)):
            layer = model.get_layer('rbm_{}'.format(i))
            layer.set_weights(
                [self.rbm_weights[i].transpose(), self.rbm_biases[i]])

        Early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=5,
                                       mode='min')

        if self.optimization is False:
            checkpointer = ModelCheckpoint(
                filepath=self.outdir +
                         "dbn_weights.hdf5",  # 给定checkpoint保存的文件名
                monitor='val_accuracy',
                mode='max',
                verbose=verbose,
                save_best_only=True)
            tensorboard = TensorBoard(
                log_dir=self.logdir)  # Tensorboard 保存地址是一个文件夹
            callbacks = [checkpointer, Early_stopping, tensorboard]
        else:
            callbacks = [Early_stopping]

        self.history = model.fit(self.data,
                                 self.targets,
                                 epochs=100,
                                 batch_size=self.batch_size,
                                 verbose=verbose,
                                 callbacks=callbacks,
                                 shuffle=True,
                                 validation_split=0.2)

        self.val_acc = self.history.history['val_accuracy']
        Error = 1 - self.val_acc[-1]

        if self.optimization is False:
            plot_learning_curve(self.history)
            print("val_Error is %f:" % Error)

        self.model = model


    def save_model(self):
        model = self.model
        model.save(self.outdir + "dbn_model")

    # def load_rbm(self):
    #     try:
    #         self.rbm_weights = pickle.load(self.rbm_dir + "rbm_weights.p")
    #         self.rbm_biases = pickle.load(self.rbm_dir + "rbm_biases.p")
    #         self.rbm_h_act = pickle.load(self.rbm_dir + "rbm_hidden.p")
    #     except:
    #         print("No such file or directory.")

    def get_score(self, params, rul_pre=False):
        """
        Get DBN error rate

        :param params: list [Dropout，leringrate_rbm,lerningrata_nn]
        :return: float, error rate
        """
        self.pretrain(params, verbose=0)
        self.finetune(params, verbose=0)
        # score = (self.val_acc[-3] + self.val_acc[-2] + self.val_acc[-1]) / 3
        # pre = self.model.predict(self.x_test)  # 对测试集进行分类预测
        # Error = 1 - score  # 计算测试分类正确率
        # return pre, Error

        # x_train = self.data
        # x_test = self.x_test
        # y_train = self.targets
        # y_test = self.y_test
        # results = params
        # self.pretrain(params, verbose=0)
        # self.finetune(params, verbose=0)
        # hist = train_model(x_train, x_test, y_train, y_test, results)

        # savePath = save_model(model, savePath)
        # final_loss = model.history['val_loss'][-1]
        # loss = (int(final_loss * 10000) / 10000)
        # print(loss)
        # accuracy = 1 - loss
        pre = self.model.predict(self.x_test)
        # y_test = y_test.T[0]

        if self.rul_pre:
            pre_use = pre.ravel()
            y_test_use = self.y_test.ravel()
            pre_use[pre_use>1] = 1
            accuracy = r2_score(y_test_use, pre_use)  # 准确率
            Error = 1 - accuracy
        else:
            pre_use = np.argmax(pre, axis=1)
            y_test_use = np.argmax(self.y_test, axis=1)
            accuracy = accuracy_score(y_test_use, pre_use)  # 准确率
            Error = 1 - accuracy

        y = (y_test_use, pre_use)

        return y, Error

