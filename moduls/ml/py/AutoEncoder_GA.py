#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from random import choice
import sys
import os

# from google.colab import drive
#
# drive.mount('/content/drive')


# os.chdir('drive/MyDrive/2020WS/GA/')
# sys.path.append('drive/MyDrive/2020WS/GA/')

# pip install import-ipynb

# import import_ipynb
import Autoencoder


# 轮盘赌选择法
def select(pop, fitness, POP_SIZE):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness / fitness.sum())
    return pop[idx]


# 交叉
def crossover(parent, pop, CROSS_RATE, POP_SIZE, DNA_SIZE, selection):
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)  # 随机取一条基因的index
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)  # 随机选取交叉点
        for i, point in enumerate(cross_points):
            if i < 6:
                if selection[2 * i] != selection[2 * i + 1]:
                    if point == True and pop[i_, i] * parent[i] == 0:  # 任一染色体上此位置上的基因为0，取消此位置的交换
                        cross_points[i] = False
                    if point == True and i < 1:  # 要交换的基因是关于层数的，取消此位置的交换
                        cross_points[i] = False
        parent[cross_points] = pop[i_, cross_points]  # 进行随机点位交叉

    return parent


# 变异
def mutate(child, DNA_SIZE, MUTATION_RATE, selection):
    k = 257
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            if point >= 1 and point <= 3:
                if child[point] != 0:
                    if selection[2 * point] != selection[2 * point + 1]:
                        child[point] = np.random.randint(32, k)
                        k = child[point]
            elif point == 4:
                if selection[2 * point] != selection[2 * point + 1]:
                    child[point] = np.random.randint(15, 30, size=1).tolist()[0]
            elif point == 5:
                if selection[2 * point] != selection[2 * point + 1]:
                    left = selection[2 * point]
                    right = selection[2 * point + 1]
                    arr1 = [32, 64, 128]
                    arr2 = []
                    for i in range(0, len(arr1)):
                        if arr1[i] >= left and arr1[i] <= right:
                            arr2.append(arr1[i])
                    child[point] = choice(arr2)
            elif point == 6:
                if selection[2 * point] != 'default':
                    child[point] = choice(['softmax', 'sigmoid', 'relu', 'tanh'])
            elif point == 7:
                if selection[2 * point - 1] != 'default':
                    child[point] = choice(['adam', 'rmsprop', 'adagrad'])
            elif point == 8:
                if selection[2 * point - 2] != 'default':
                    child[point] = choice(['mse', 'mae'])
    return child


def create_initial_data(selection, data):  # data是x_train,y_train,x_value,y_value
    pop_size = selection[15]  # pop_size是for循环的次数，表示要生成多少组数据
    pop = np.empty([pop_size, 9], dtype=object)  # pop装所有要优化的参数
    fitness = np.zeros([pop_size, ])  # 用fitness装每组参数在LSTM训练后的accuracy
    for i in range(0, pop_size):
        results = []
        LayerCount = 0
        if selection[0] == selection[1]:
            LayerCount = selection[0]
        else:
            LayerCount = np.random.randint(selection[0], selection[1], size=1).tolist()[0]
        units1 = 0
        units2 = 0
        units3 = 0
        if LayerCount == 1:
            if selection[2] == selection[3]:
                units1 = selection[2]
            else:
                units1 = np.random.randint(selection[2], selection[3], size=1).tolist()[0]
        elif LayerCount == 2:
            if selection[2] == selection[3]:
                units1 = selection[2]
            else:
                units1 = np.random.randint(selection[2], selection[3], size=1).tolist()[0]

            if selection[4] == selection[5]:
                units2 = selection[4]
            else:
                units2 = np.random.randint(selection[4], selection[5], size=1).tolist()[0]

        else:
            if selection[2] == selection[3]:
                units1 = selection[2]
            else:
                units1 = np.random.randint(selection[2], selection[3], size=1).tolist()[0]

            if selection[4] == selection[5]:
                units2 = selection[4]
            else:
                units2 = np.random.randint(selection[4], selection[5], size=1).tolist()[0]

            if selection[6] == selection[7]:
                units3 = selection[6]
            else:
                units3 = np.random.randint(selection[6], selection[7], size=1).tolist()[0]

        epochs = 0
        if selection[8] == selection[9]:
            epochs = selection[8]
        else:
            epochs = np.random.randint(selection[8], selection[9], size=1).tolist()[0]

        batchSize = 0
        if selection[10] == selection[11]:
            batchSize = selection[10]
        else:
            left = selection[10]
            right = selection[11]
            arr1 = [32, 64, 128]
            arr2 = []
            for i in range(0, len(arr1)):
                if arr1[i] >= left and arr1[i] <= right:
                    arr2.append(arr1[i])
            batchSize = choice(arr2)

        denseActivation = 0
        if selection[12] == 'default':
            denseActivation = choice(['softmax', 'sigmoid', 'relu', 'tanh'])
        else:
            denseActivation = selection[12]

        optimizer = 0
        if selection[13] == 'default':
            optimizer = choice(['adam', 'rmsprop', 'adagrad'])
        else:
            optimizer = selection[13]

        loss = 0
        if selection[14] == 'default':
            loss = choice(['mse', 'mae', 'categorical_crossentropy'])
        else:
            loss = selection[14]
        # results[LayerCount,units1,units2,units3,epochs,batchSize,denseActivation,optimizer,loss]
        results.append(LayerCount)
        results.append(units1)
        results.append(units2)
        results.append(units3)
        results.append(epochs)
        results.append(batchSize)
        results.append(denseActivation)
        results.append(optimizer)
        results.append(loss)
        pop[i] = results
        fitness[i] = get_fitness(results, data)
        print('第' + str(i + 1) + '个DNA')
    return pop, fitness  # pop+finess就相当于我们自己生成的初始数据


def load_data(path1, path2):
    train_data = np.array(pd.read_csv(path1))  # 读取训练数据
    test_data = np.array(pd.read_csv(path2))  # 读取训练数据
    shape = train_data.shape[1]  #
    data = shape, train_data, test_data
    return data


def get_fitness(results, data):
    return Autoencoder.get_accuracy(results, data)


def run(dna_size, pop_size, cross_rate, mutation_rate, n_generations, path1, path2, selection, threshold):
    DNA_SIZE = dna_size
    POP_SIZE = pop_size
    CROSS_RATE = cross_rate
    MUTATION_RATE = mutation_rate
    N_GENERATIONS = n_generations

    # 读取数据
    # data = load_data(path1, path2) #(shape, train_data, test_data)
    from dataset import import_data
    x_train, x_test, y_train, y_test = import_data("../Dataset/")
    shape = x_train.shape[1]
    data = shape, x_train, x_test

    ret = create_initial_data(selection, data)
    pop = ret[0]
    fitness = ret[1]
    original_pop = pop
    original_fitness = fitness
    print(pop)
    print(fitness)

    MAX_Index = np.argmax(fitness)  # 找出初始种群最优 5
    MAX_DNA = pop[MAX_Index, :]  # [....]

    fitness = fitness.reshape(POP_SIZE, )

    best_generation = []

    flag = 0
    dna_set = []
    for each_generation in range(N_GENERATIONS):

        pop = select(pop, fitness, POP_SIZE)  # 选择
        pop_copy = pop.copy()
        for parent in pop:
            child = crossover(parent, pop, CROSS_RATE, POP_SIZE, DNA_SIZE, selection)  # 交叉
            child = mutate(child, DNA_SIZE, MUTATION_RATE, selection)  # 变异
            parent = child

        if fitness[0] <= fitness[MAX_Index]:  # 如果第一个DNA适应度比最大的小，则将MAX_DNA放置首位
            pop[0] = MAX_DNA
        fitness = np.zeros([POP_SIZE, ])
        for i in range(POP_SIZE):
            pop_list = list(pop[i])
            fitness[i] = get_fitness(pop_list, data)
            print('第%d代第%d个染色体的适应度为%f' % (each_generation + 1, i + 1, fitness[i]))
            print('此染色体为：', pop_list)
            if fitness[i] >= threshold:
                flag = 1
                print("条件终止")
                break
        print("Generation:", each_generation + 1, "Most fitted DNA: ", pop[np.argmax(fitness), :], "适应度为：",
              fitness[np.argmax(fitness)])
        MAX_Index = np.argmax(fitness)
        MAX_DNA = pop[MAX_Index, :]
        dna_set.append(MAX_DNA)
        best_generation.append(fitness[MAX_Index])
        if flag == 1:
            break
    return best_generation, dna_set, original_pop, original_fitness


def fitness_figure(best_generation):
    x = np.linspace(1, len(best_generation), len(best_generation))
    y = best_generation
    plt.plot(x, y)
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.show()


if __name__ == '__main__':

    path1 = '/content/drive/MyDrive/2020WS/GA/traindata.csv'
    path2 = '/content/drive/MyDrive/2020WS/GA/testdata.csv'
    selection = [1, 4, 210, 250, 160, 209, 100, 155, 20, 20, 128, 128, 'default', 'default', 'mse', 30]
    threshold = 0.9960
    best_generation, dna_set, original_pop, original_finess = run(9, 30, 0.3, 0.1, 7, path1, path2, selection,
                                                                  threshold)
    print(best_generation)
    print(dna_set)
    print(original_pop)
    print(original_finess)

    fitness_figure(best_generation)

    best_generation = [0.9506, 0.97, 0.9936, 0.9936, 0.9936, 0.9936, 0.9955]
    x = np.linspace(1, len(best_generation), len(best_generation))
    y = best_generation
    plt.plot(x, y)
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.show()
