from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import scipy.io as sio
import numpy as np
import os
from moduls.feature_selection.writein import writein

from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func

def AE(data, encoding_dim=4,save_path='./',output_file=0):
    '''

    :param data: input features,mxn
    :param switch: 0:features in raw,1:features in column
    :param encoding_dim:dimension after reduction
    :param save_path:path to save
    :param output_file:type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :return:
    '''
    # Normalise
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Fixed dimensions
    input_dim = data.shape[1] # 8
    if encoding_dim>input_dim or encoding_dim <1:
        abort(400,'The feature dimension must be in [1, %d].'% input_dim)
    # encoding_dim = 3
    # Number of neurons in each Layer [8, 6, 4, 3, ...] of encoders
    input_layer = Input(shape=(input_dim, ))
    encoder_layer = Dense(encoding_dim, activation="tanh")(input_layer)

    # Crear encoder model

    encoder = Model(inputs=input_layer, outputs=encoder_layer)
    # Use the model to predict the factors which sum up the information of interest rates.
    encoded_data = pd.DataFrame(encoder.predict(data_scaled))
    encoded_data= encoded_data.values

    # 保存数据
    save_data_func(data=encoded_data, output_file=output_file, save_path=save_path,
                   file_name="AE",
                   index_label="Downscaled data")

    return encoded_data

if __name__=="__main__":
    # data = writein('time_features.mat')
    # data = writein('x_train_feature.mat')
    data = writein('traindata.mat',1)
    print(data.shape)
    encoded_data=AE(data)
    print(encoded_data.shape)