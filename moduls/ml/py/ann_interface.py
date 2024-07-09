import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout
from sklearn.model_selection import train_test_split

# 常量设定
BATCH_SIZE = 32
EPOCHS = 100


class paramANN:
    """
    This interface is used for User-defined parameters of ANN
    # User can define the parameters(units,activation functions) through this interface
    # units of each layers
    # activation function for Dense layers
    
    # different from LSTM, the layers in ANN are all Dense layers,
    # therefore, we have no need to create the paramDense interface
    该接口用于人工神经网络的自定义参数
     # 用户可以通过这个接口定义参数（单位，激活函数）
     # 每层的单位
     # 密集层的激活函数

     # 与 LSTM 不同的是，ANN 中的层都是 Dense 层，
     # 因此，我们不需要创建 paramDense 接口
    """
    units = 5
    activation = 'relu'
    recurrent_activation = 'sigmoid'
    use_bias = True
    kernel_initializer = 'glorot_uniform'
    recurrent_initializer = 'orthogonal'
    bias_initializer = 'zeros'
    unit_forget_bias = True
    kernel_regularizer = None
    recurrent_regularizer = None
    bias_regularizer = None
    activity_regularizer = None
    kernel_constraint = None
    recurrent_constraint = None
    bias_constraint = None
    dropout = 0.0
    recurrent_dropout = 0.0
    implementation = 2
    return_sequences = True
    return_state = False
    go_backwards = False
    stateful = False
    time_major = False
    unroll = False


class paramDropout:
    """
    This interface is used for User-defined Dropoutrate of ANN, closely similar to LSTM
    # User can define the Dropout rate through this interface
    # If Dropout rate equals, then there won't be dropout layers
    # Dropout rate
    该接口用于ANN的User-defined Dropoutrate，与LSTM非常相似
     # 用户可以通过该接口定义Dropout率
     # 如果 Dropout rate 相等，则不会有 dropout 层
     # 辍学率
    """
    rate = 0.2
    noise_shape = None
    seed = None


class compileOption:
    """
    This interface is used for User-defined parameters(optimizer,loss) through this interface
    #  Optimizer: possible choices includes: "adam", "adagrad", "rmsprop", etc.
    #  Loss: possible choices includes: "mse", "binary_crossentropy", "mae",etc.
    该接口用于通过该接口自定义参数（优化器、损失）
     # 优化器：可能的选择包括：“adam”、“adagrad”、“rmsprop”等。
     # Loss：可能的选择包括：“mse”、“binary_crossentropy”、“mae”等。
    """
    # TODO:超参数：Optimizer，Loss
    optimizer = "rmsprop"
    loss = 'mse'
    metrics = ['accuracy']
    loss_weights = None,
    weighted_metrics = None
    run_eagerly = None


class fitOption:
    """
    This interface is used for User-defined fit options through this interface
    # User can define the epochs & batchsize
    该接口用于通过该接口自定义拟合选项
     # 用户可以定义 epochs & batchsize
    """
    # TODO:超参数，x,y/epochs & batchsize
    x = None
    y = None
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    verbose = 1
    callbacks = None
    validation_split = 0.0
    validation_data = None
    shuffle = True
    class_weight = None
    sample_weight = None
    initial_epoch = 0
    steps_per_epoch = None
    validation_steps = None
    validation_batch_size = None
    validation_freq = 1
    max_queue_size = 10
    workers = 1
    use_multiprocessing = False


def fun_add(typ, Param, model):
    D(typ, Param, model)


def D(typ, Param, model):
    """
    We only use the Dense layers to build the ANN model
    我们只使用 Dense 层来构建 ANN 模型
    """
    model.add(tf.keras.layers.Dense(Param.units
                                    , activation=Param.activation
                                    , use_bias=Param.use_bias
                                    , kernel_initializer=Param.kernel_initializer
                                    , bias_initializer=Param.bias_initializer
                                    , kernel_regularizer=Param.kernel_regularizer
                                    , bias_regularizer=Param.bias_regularizer
                                    , activity_regularizer=Param.activity_regularizer
                                    , kernel_constraint=Param.kernel_constraint
                                    , bias_constraint=Param.bias_constraint
                                    )
              )


def fun_compile(model, compileOption):
    """
    This interface is to run compile the model through all the options 
    configured by the users, namely the compile options
    该界面是通过用户配置的所有选项运行编译模型，即编译选项
    """
    model.compile(optimizer=compileOption.optimizer
                  , loss=compileOption.loss
                  , metrics=compileOption.metrics
                  , loss_weights=compileOption.loss_weights
                  , weighted_metrics=compileOption.weighted_metrics
                  , run_eagerly=compileOption.run_eagerly)


def fun_fit(model, fitOption):
    """
    Similarly, this interface is to fit the model through all the options 
    configured by the users, namely the fit options such as batchsize, epochs
    同样，这个接口是通过用户配置的所有选项来拟合模型，即batchsize、epochs等拟合选项
    """
    model.fit(fitOption.x
              , fitOption.y
              , batch_size=fitOption.batch_size
              , epochs=fitOption.epochs
              , verbose=fitOption.verbose
              , callbacks=fitOption.callbacks
              , validation_split=fitOption.validation_split
              , validation_data=fitOption.validation_data
              , shuffle=fitOption.shuffle
              , class_weight=fitOption.class_weight
              , sample_weight=fitOption.sample_weight
              , initial_epoch=fitOption.initial_epoch
              , steps_per_epoch=fitOption.steps_per_epoch
              , validation_steps=fitOption.validation_steps
              , validation_batch_size=fitOption.validation_batch_size
              , validation_freq=fitOption.validation_freq
              , max_queue_size=fitOption.max_queue_size
              , workers=fitOption.workers
              , use_multiprocessing=fitOption.use_multiprocessing
              )


def train_model(Shape, typList, ParamList, compileOption, fitOption,
                batch_size=BATCH_SIZE, epochs=EPOCHS):
    tf.random.set_seed(42)
    """
    This interface is used to build the ANN model, in the environment of 
    Tensorflow keras, generally we build the sequential model of ANN
    该接口用于构建ANN模型，在Tensorflow keras环境下，一般我们构建ANN的序列模型
    """
    # build the ANN model
    model = Sequential()
    model.add(Input(shape=Shape))
    length = len(typList)  # typList=['Dense','Dense','Dense']
    for i in range(length):  # ParamList=[paramANN(),paramANN(),paramDense()]
        fun_add(typList[i], ParamList[i], model)
        # fun_add(ParamList[i], model)
    fun_compile(model, compileOption)

    # print the model summary to users      向用户打印模型摘要
    model.summary()
    fun_fit(model, fitOption)

    return model


def evaluate_model(model, x_val, y_val, batch_size=32):
    """
    model evaluation with the x_val,y_val given by the users
    使用用户给出的 x_val,y_val 进行模型评估
    """
    print(model.evaluate(x_val, y_val, batch_size, verbose=1))
    print(model.metrics_names)
    # loss, accuracy = model.evaluate(x_val, y_val, batch_size, verbose=1)
    loss, accuracy= model.evaluate(x_val, y_val, batch_size, verbose=1)
    return loss, accuracy


def read_data(path_to_data):
    import scipy.io

    train_data = scipy.io.loadmat(path_to_data + "traindata.mat")  # 读取mat文件
    train_label = scipy.io.loadmat(path_to_data + "trainlabel.mat")
    test_data = scipy.io.loadmat(path_to_data + "testdata.mat")  # 读取mat文件
    test_label = scipy.io.loadmat(path_to_data + "testlabel.mat")
    # print(train_data.keys())  # 查看mat文件中的所有变量
    # print(train_data['__header__'])
    # print(train_data['__version__'])
    # print(train_data['__globals__'])
    # print(train_label['train_label'])
    x_train = train_data['train_data']
    y_train = train_label['train_label']
    x_test = test_data['test_data']
    y_test = test_label['test_label']

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':

    # 读入数据

    path_to_data = "./Dataset/"
    x_train, y_train, x_test, y_test = read_data(path_to_data)

    # 数据准备
    typ_list = ['Dense', 'Dense', 'Dense']
    param_list = [paramANN(), paramANN(), paramANN()]
    compile_option = compileOption
    fit_option = fitOption

    fit_option.x = x_train
    fit_option.y = y_train

    model = train_model(x_train.shape, typ_list, param_list, compile_option, fit_option)

    # 模型评估(模型对象，)
    evaluate_model(model, x_test, y_test, 32)
