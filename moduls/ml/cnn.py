"""Convolutional Neural Network"""
import os
import shutil
from tensorflow import keras
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
from moduls.ml.utils import plot_learning_curve
from moduls.ml.utils import translate_params


def build_network(dropout=0.5, learning_rate=0.004, num_conv=6):
    """
    Build the CNN network.

    :param num_conv:
    :param learning_rate:
    :param dropout:
    :return: CNN model built from given parameters
    """
    # build model from given parameters
    model = keras.models.Sequential()
    # model.add(keras.layers.Conv1D(filters=16, kernel_size=8, activation='relu',
    #                               input_shape=(576, 1)))
    model.add(keras.layers.Conv1D(filters=16, kernel_size=8, activation='relu',
                                  input_shape=(38, 1)))
    model.add(keras.layers.Conv1D(filters=16, kernel_size=8, activation='relu', padding="same"))
    if num_conv >= 4:
        model.add(keras.layers.MaxPooling1D(2))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', padding="same"))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', padding="same"))
    if num_conv >= 6:
        model.add(keras.layers.MaxPooling1D(2))
        model.add(keras.layers.Conv1D(filters=256, kernel_size=2, activation='relu', padding="same"))
        model.add(keras.layers.Conv1D(filters=256, kernel_size=2, activation='relu', padding="same"))
    if num_conv >= 8:
        model.add(keras.layers.MaxPooling1D(2))
        model.add(keras.layers.Conv1D(filters=512, kernel_size=1, activation='relu', padding="same"))
        model.add(keras.layers.Conv1D(filters=512, kernel_size=1, activation='relu', padding="same"))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dropout(dropout))
    # model.add(keras.layers.Dense(4, activation='softmax'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=opt,
    #               metrics='accuracy')
    model.compile(loss='mse',
                  optimizer=opt,
                  metrics='accuracy')
    return model


class CNN_Model:

    def __init__(self, x_train, y_train,
                 x_test, y_test,
                 rul_pre=False,
                 discrete_candidate=None,
                 outdir=None,
                 logdir=None,
                 optimization=True,
                 epoch=5):
        """
        :param x_train: training data
        :param y_train: training label
        :param x_test: test data
        :param y_test: test label
        :param outdir: output directory
        :param logdir: log directory
        :param optimization: Bool, is it for optimization
        :param epoch: int, training epoch number
        """
        # print('Selected Network : CNN')
        self.optimization = optimization
        self.epoch = epoch
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = None
        self.rul_pre = rul_pre
        # if self.optimization is False:
        #     if os.path.exists(outdir):
        #         shutil.rmtree(outdir)
        #     if os.path.exists(logdir):
        #         shutil.rmtree(logdir)
        #     os.makedirs(outdir)
        #     os.makedirs(logdir)
        #
        #     self.outdir = outdir
        #     self.logdir = logdir
        self.discrete = discrete_candidate

    def build_network(self, dropout=0.5, learning_rate=0.004, num_conv=6):
        """
        Build the CNN network.

        :param num_conv:
        :param learning_rate:
        :param dropout:
        :return: CNN model built from given parameters
        """

        if not self.rul_pre:
            print("创建分类模型", self.rul_pre)
            # build model from given parameters
            model = keras.models.Sequential()
            # model.add(keras.layers.Conv1D(filters=16, kernel_size=8, activation='relu',
            #                               input_shape=(576, 1)))
            model.add(keras.layers.Conv1D(filters=16, kernel_size=8, activation='relu',
                                          input_shape=(self.x_train.shape[1], 1)))
            model.add(keras.layers.Conv1D(filters=16, kernel_size=8, activation='relu', padding="same"))
            if num_conv >= 4:
                model.add(keras.layers.MaxPooling1D(2))
                model.add(keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', padding="same"))
                model.add(keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', padding="same"))
            if num_conv >= 6:
                model.add(keras.layers.MaxPooling1D(2))
                model.add(keras.layers.Conv1D(filters=256, kernel_size=2, activation='relu', padding="same"))
                model.add(keras.layers.Conv1D(filters=256, kernel_size=2, activation='relu', padding="same"))
            if num_conv >= 8:
                model.add(keras.layers.MaxPooling1D(2))
                model.add(keras.layers.Conv1D(filters=512, kernel_size=1, activation='relu', padding="same"))
                model.add(keras.layers.Conv1D(filters=512, kernel_size=1, activation='relu', padding="same"))
            model.add(keras.layers.GlobalAveragePooling1D())
            model.add(keras.layers.Dropout(dropout))
        # if not self.rul_pre:
            model.add(keras.layers.Dense(self.y_train.shape[1], activation='softmax'))
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy']
                          )
        else:
            print("创建回归模型", self.rul_pre)
            model = keras.models.Sequential()
            # model.add(keras.layers.Conv1D(filters=16, kernel_size=8, activation='relu',
            #                               input_shape=(576, 1)))
            model.add(keras.layers.Conv1D(filters=16, kernel_size=7, activation='softplus',
                                          input_shape=(self.x_train.shape[1], 1)))
            model.add(keras.layers.MaxPooling1D(2))
            model.add(keras.layers.Conv1D(filters=16, kernel_size=7, activation='sigmoid', padding="same"))
            if num_conv >= 4:
                model.add(keras.layers.MaxPooling1D(2))
                model.add(keras.layers.Conv1D(filters=64, kernel_size=5, activation='sigmoid', padding="same"))
                model.add(keras.layers.Conv1D(filters=64, kernel_size=5, activation='sigmoid', padding="same"))
            if num_conv >= 6:
                model.add(keras.layers.MaxPooling1D(2))
                model.add(keras.layers.Conv1D(filters=256, kernel_size=3, activation='sigmoid', padding="same"))
                model.add(keras.layers.Conv1D(filters=256, kernel_size=3, activation='sigmoid', padding="same"))
            if num_conv >= 8:
                model.add(keras.layers.MaxPooling1D(2))
                model.add(keras.layers.Conv1D(filters=512, kernel_size=1, activation='sigmoid', padding="same"))
                model.add(keras.layers.Conv1D(filters=512, kernel_size=1, activation='sigmoid', padding="same"))
            model.add(keras.layers.GlobalAveragePooling1D())
            model.add(keras.layers.Dropout(dropout))
            model.add(keras.layers.Dense(self.y_train.shape[1], activation='sigmoid'))
            opt = keras.optimizers.Adam(learning_rate=learning_rate)

            model.compile(loss='mse',
                          optimizer=opt,
                          metrics=['accuracy']
                          )
        return model

    def train(self, params, plot=False):
        """
        Build and train the CNN, use the given parameters.

        :param params: list, [dropout, learning_rate, batch_size]
        :param plot: bool, plot the learning curve
        :return: training history and the model for evaluate
        """
        # call the building function
        if self.optimization:
            dropout, learning_rate, batch_size, conv = translate_params(params, self.discrete)
        else:
            dropout = params[0]
            learning_rate = params[1]
            batch_size = params[2]
            conv = params[3]
        self.model = self.build_network(dropout=dropout,
                                        learning_rate=learning_rate,
                                        num_conv=conv)

        Early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=3,
                                                       mode='min')
        verbose = 1
        if self.optimization is False:
            verbose = 1
            checkpointer = keras.callbacks.ModelCheckpoint(filepath=self.outdir + "cnn_weights.hdf5",
                                                           monitor='val_accuracy',
                                                           mode='max',
                                                           verbose=verbose,
                                                           save_best_only=True)
            tensorboard = keras.callbacks.TensorBoard(log_dir=self.logdir)  # Tensorboard 保存地址是一个文件夹
            callbacks = [checkpointer, Early_stopping, tensorboard]
        else:
            callbacks = [Early_stopping]

        self.model.summary()
        history = self.model.fit(self.x_train,
                                 self.y_train,
                                 batch_size=batch_size,
                                 callbacks=callbacks,
                                 epochs=self.epoch,
                                 verbose=verbose,
                                 validation_split=0.3)
        if plot:
            plot_learning_curve(history)
        return history

    def cnn_get_error(self, params):
        """
        Function to get the score of each model.

        :param params: list, [dropout, learning_rate, batch_size, conv numbers]
        :return: float, 1- mean value from last 3 validation accuracy
        """
        history = self.train(params)

        if not self.rul_pre:
            print("history.history", history.history.keys())
            # val_acc = history.history['val_accuracy']
            val_acc = history.history['acc']
            score = (val_acc[-3] + val_acc[-2] + val_acc[-1]) / 3  # choose average of last three accuracy
            error = 1 - score
            pre = history.model.predict(self.x_test)
        else:
            pre = history.model.predict(self.x_test)
            score = r2_score(self.y_test, pre)
            if score < 0:
                score = 0.01
            error = 1 - score
        print("成绩:", score)
        pre_use = np.argmax(pre, axis=1)
        y_test_use = np.argmax(self.y_test, axis=1)
        return error

    def get_score(self, params):
        history = self.train(params)

        if not self.rul_pre:
            print("history.history", history.history.keys())
            # val_acc = history.history['val_accuracy']
            val_acc = history.history['acc']
            score = (val_acc[-3] + val_acc[-2] + val_acc[-1]) / 3  # choose average of last three accuracy

            error = 1 - score
            pre = history.model.predict(self.x_test)
            print("pre", pre)
            pre_use = np.argmax(pre, axis=1)
            y_test_use = np.argmax(self.y_test, axis=1)
        else:
            pre = history.model.predict(self.x_test)
            print("故障诊断结果:", pre)
            score = r2_score(self.y_test, pre)
            if score < 0:
                score = 0.01
            error = 1 - score
            pre_use = pre.reshape(-1)
            y_test_use = self.y_test.reshape(-1)
        print("成绩:", score)


        print("pre", pre)
        print("self.y_test", self.y_test)

        y = (y_test_use, pre_use)

        return y, error

    def test(self):
        """
        Evaluate the model

        :return: None
        """
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        print("Test accuracy:", test_acc)

    def save_model(self):
        """
        Save cnn model public Function

        :return: None
        """
        self.model.save(self.outdir + "cnn_model")

    def report(self, data, labels):
        """
        Generating network report

        :param data: training data
        :param labels: training label
        :return: None
        """
        print(classification_report(np.argmax(labels, axis=1),
                                    np.argmax(self.model.predict(data), axis=1),
                                    digits=4))


"""
Demo-code for CNN testing:
-------------------------------------------
x_train, x_test, y_train, y_test = import_data("./Dataset/", model='CNN')
cnn = CNN(x_train, y_train, x_test, y_test,
          outdir="./model/",
          logdir="./log/",
          optimization=False,
          epoch=10)
history = cnn.train([0.64, 0.004, 16, 6], plot=True)
cnn.report()
-------------------------------------------
Suggestion:
1. first argument for cnn.train should be: 
    [dropout, learning_rate, batch size, number of conv(4, 6, 8)]
2. try bigger batch_size
"""

