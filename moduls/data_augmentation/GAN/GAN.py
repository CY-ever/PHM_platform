#!/usr/bin/env python
# coding: utf-8

"""
Created on Apr 9 2021
This script contains functions to perform data augmentation by GAN.
Function writein loads data from input loadpath of .mat/.npy/.xls/.xlsx/.txt/.csv file. The file contains horizontal vibration signal of bearing at first column and header at first row. Function has following input:
    loadpath: loadpath of input data(string)
Function fre returns fault characteristic frequency, with the following inputs:
    faulttype: type of fault, 0 for FTF, 1 for BPFO, 2 for BPFI
    rot_fre: frequency of rotation in test
    n_ball: the number of rolling elements
    d_ball: the ball diameter
    D_pitch: the pitch diameter
    alpha: the bearing contact angle
Function timeplot draws time-domain diagram, with the following inputs:
    data: input data with shape (n,)
    fs: sampling frequency
Function envspectrum and envspectrumplot draws Hilbert envelope spectrum, with the following inputs:
    data: input data with shape (n,)
    fs: sampling frequency
    f_fault: fault characteristic frequency
    order: order of fault characteristic frequency
Function cosine_similarity draws frequency-domain diagram, with the following inputs:
    x,y: input data with shape (n,)
Function GAN_DA performs data augmentatio and returns new data after augmentation, cosine-similarity, the trend of cosine-similarity, discriminator-loss and generator-loss, with the following inputs:
    num: the number of the increased dataset
    numEpochs: the number of iterations in GAN training
    Z_dim: size of input noise, which is used to generate new data
@author: Jin Wang/ Yifan Zhao
version: V1.1
"""

# import tensorflow as tf
# import tensorflow.compat.v1 as tf
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import hilbert
import math
import os
import scipy.io as sio
import xlrd
import pandas as pd


def envspectrum(data):
    N = len(data)
    hilbert1 = np.abs(hilbert(data))
    Y = np.abs(fft(hilbert1)) / N
    Y[0] = 0
    return Y


def envspectrumplot(data, f_fault, fs, order):
    N = len(data)
    Y = envspectrum(data)
    f = np.arange(N) * fs / N

    plt.figure(figsize=(20, 4))
    plt.plot(f, Y)
    faultline = np.array([f_fault, f_fault])
    for i in range(order):
        plt.plot(faultline * (i + 1), [0, 2], linestyle="--", color="orange")
    plt.xlabel('Frequency(Hz)')
    plt.xlim(0, 800)
    plt.ylim(0, 2.5)
    plt.title("Envelope spectrum")
    # plt.show()


def cosine_similarity(x, y):
    num = np.dot(x, y)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom


def GAN_DA(data_use, num, Z_dim):
    # initialization of variable
    def variable_init(size):
        in_dim = size[0]
        w_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=w_stddev)

    # define inputs, weights and bias of network
    # X = tf.placeholder(tf.float32, shape=[32761])
    X = tf.placeholder(tf.float32, shape=None)
    Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
    # D_W1 = tf.Variable(variable_init([32761, 100]))
    D_W1 = tf.Variable(variable_init([len(data_use), 100]))
    D_b1 = tf.Variable(tf.zeros(shape=[100]))
    D_W2 = tf.Variable(variable_init([100, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))
    theta_D = [D_W1, D_W2, D_b1, D_b2]
    G_W1 = tf.Variable(variable_init([Z_dim, 400]))
    G_b1 = tf.Variable(tf.zeros(shape=[400]))
    # G_W2 = tf.Variable(variable_init([400, 32761]))
    G_W2 = tf.Variable(variable_init([400, len(data_use)]))
    # G_b2 = tf.Variable(tf.zeros(shape=[32761]))
    G_b2 = tf.Variable(tf.zeros(shape=[len(data_use)]))
    theta_G = [G_W1, G_W2, G_b1, G_b2]

    # generate noise as input of generator
    def sample_Z(m, n):
        return np.random.uniform(-1., 1., size=[m, n])

    # build generator
    def generator(z):
        G_h1 = tf.nn.leaky_relu(tf.matmul(z, G_W1) + G_b1)
        G_h2 = (tf.matmul(G_h1, G_W2) + G_b2)
        return G_h2

    # build discriminator
    def discriminator(x):
        D_h1 = tf.nn.leaky_relu(tf.matmul(tf.expand_dims(x, 0), D_W1) + D_b1)
        D_logit = tf.matmul(D_h1, D_W2) + D_b2
        return D_logit

    # generate fake signal, use discriminator to test original and fake signal
    G_sample = generator(Z)
    D_logit_real = discriminator(X)
    D_logit_fake = discriminator(G_sample[0])

    # build loss function
    D_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    # D_loss = D_loss_real + D_loss_fake
    # G_loss = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

    # optimizer
    # D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    # G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    # load original data

    seed = sample_Z(100, Z_dim)

    Similarity = np.zeros(100)
    Simi_trend = {}
    Gloss_trend = []
    Dloss_trend = []

    # start training
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # generate new data after training
    newdata = sess.run(G_sample, feed_dict={Z: sample_Z(num, Z_dim)})
    Similarity = np.zeros(num)
    for i in range(num):
        Similarity[i] = cosine_similarity(data_use, newdata[i])
    return newdata




