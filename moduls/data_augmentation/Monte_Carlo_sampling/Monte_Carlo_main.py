#!/usr/bin/env python
# coding: utf-8

"""
This script contains functions to perform data augmentation by data fitting and Monte Carlo sampling.
Function func1-6 returns different forms of typical function.
Function writein returns input data in different file formats(.mat/.txt/.npy/.xls/.xlsx), with the following input:
    loadpath: loadpath of input data set (string)
Function func_fit returns parameters of the fitting model and threshhold, with the following input and output:
    data: inputdata of original degration trajectories
    function: the form of the function
    p:initial value of parameters
    parameter: parameters of model in form of M*N array, M represents the index of trajectory, N represents the number of parameters in function
    thresh: threshold of the estimation
Function monte_carlo returns new parameter after Monte Carlo sampling, with the following input:
    dataset: inputdata of parameters
    dis: different types of distributions, 0/weibull distribution, 1/normal distribution, 2/gamma distribution
    m: the number of the aging model library
Function Monte_Carlo_DA performs data augmentation with the following input and output:
    mode: 0 for data fitting and Monte Carlo sampling of fitted parameters, 1 for Monte Carlo sampling of input parameter
    loadpath: loadpath of folder containing input data for mode 0, data of each trajectory should be saved as a file;
           loadpath of input parameter for mode 1
    distribution: different types of distributions, 0/weibull distribution, 1/normal distribution, 2/gamma distribution
    function: the form of the function, default is func1(exponential degradation model), user-defined function must take first parameter as independent variable
    m: the number of the increased data set
    p0: initial value of parameters
    thresh: threshold of the estimation
    new_parameter: m*N array of new parameter for mode 0, N represents the number of parameters in function, newparameter[0] represents the first set of new parameters; list of new parameter for mode 1
@author: Jin Wang/ Yifan Zhao
version: V1.1
"""


import random
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import os
import xlrd
from scipy import io
from scipy import stats
from scipy.optimize import curve_fit
from sympy.abc import x
import pandas as pd
import scipy.io as sio
from flask import abort
from utils.table_setting import *
from utils.save_data import save_data_func
import xlrd
from moduls.data_augmentation.writein import writein

from flask import abort


def func1(t, a, b):
    return a * np.exp(b * t) - 1


def func2(t, a, b, c):
    return a * np.exp(b * t) + c


def func3(t, a, b, c, d, e):
    return a * np.exp(b * t) + c * np.exp(d * t) + e


def func4(t, a, b, c):
    return a * pow(t, 2) + b * t + c


def func5(t, a, b, c, d):
    return a * pow(t, 3) + b * pow(t, 2) + c * t + d


def func6(t, a, b, c, d):
    return a * np.exp(b * t) + c * pow(t, 2) + d


def Monte_writein(loadpath):
    error = 0
    dacl = os.path.splitext(loadpath)[1]
    if '.mat' in dacl:
        mat = io.loadmat(loadpath)
        keys = list(mat.keys())
        data = mat[keys[3]].flatten()
    elif '.npy' in dacl:
        data = np.load(loadpath)
    elif '.xls' in dacl:
        sheet_book = xlrd.open_workbook(loadpath)
        sheet = sheet_book.sheet_by_index(0)
        resArray = []
        for i in range(sheet.nrows):
            line = sheet.row_values(i)
            resArray.append(line)
        data = np.array(resArray).flatten()
    elif '.txt' in dacl:
        data = np.loadtxt(loadpath)
    elif '.csv' in dacl:
        pd_reader = pd.read_csv(loadpath)
        data1 = pd_reader['acc']
        data = list(data1)
    else:
        data = None
        error = 1
    # print(data)
    return data, error


def func_fit(data, function, p, save_path, output_image=0):
    parameter = np.zeros([len(data), function.__code__.co_argcount - 1])
    thresh = 0
    plt.figure(figsize=(8, 5))
    for i in range(len(data)):
        HI = data[i]
        thresh += HI[-1]
        time = np.arange(len(HI))
        popt, pcov = curve_fit(function, time, HI, p0=p, maxfev=5000000)
        # popt, pcov = curve_fit(function, time, HI, p0=p,bounds=([0.1,0.1,0.1,0.1,0.1], [np.inf,np.inf,np.inf,np.inf,np.inf]), maxfev=5000)
        # popt, pcov = curve_fit(function, time, HI,bounds = ([-np.inf,0,0,0,0], [np.inf,1,1,1,1]), p0=p, maxfev=5000)
        for a in range(function.__code__.co_argcount - 1):
            parameter[i][a] = popt[a]

        yvals = function(time, *parameter[i])
        plt.plot(yvals, linewidth=0.5)
        plt.title('Degradation trajectories after fitting', fontsize=16)
        plt.xlabel('Time/min')
        plt.ylabel('Health Indicator')

    if output_image == 0:
        file_name = "%s.png" % "Degradation_trajectories_after_fitting"
    elif output_image == 1:
        file_name = "%s.jpg" % "Degradation_trajectories_after_fitting"
    elif output_image == 2:
        file_name = "%s.svg" % "Degradation_trajectories_after_fitting"
    elif output_image == 3:
        file_name = "%s.pdf" % "Degradation_trajectories_after_fitting"

    path = os.path.join(save_path, file_name)
    plt.savefig(path)
    # plt.show()
    plt.close()

    thresh = thresh / len(data)
    return parameter, thresh


def monte_carlo(dataset, dis=0, m=100):
    sample = stats.uniform.rvs(size=m)
    new_par = []
    if dis == 1:
        par = stats.norm.fit(dataset)
        for i in range(m):
            new_par.append(stats.norm.ppf(sample[i], par[0], par[1]))
    elif dis == 2:
        # dataset1 = [i + 10*np.e**120 for i in dataset]
        par = stats.gamma.fit(dataset, floc=0)
        for i in range(m):
            new_par.append(stats.gamma.ppf(sample[i], par[0], 0, par[2]))
    elif dis == 0:
        par = stats.weibull_min.fit(dataset, floc=0)
        for i in range(m):
            new_par.append(stats.weibull_min.ppf(sample[i], par[0], 0, par[2]))
    return new_par


# 主函数
def Monte_Carlo_DA(loadpath, save_path='./', mode=0, distribution=0, function_select=0, m=50, a=0.05, b=0.003, c=0,
                   d=0.003, e=-0.001, output_file=0, output_image=0):
    '''

    :param loadpath: loadpath of folder containing input data for mode 0, data of each trajectory should be saved as a file;
           loadpath of input parameter for mode 1
    :param save_path:path to save
    :param mode:0 for data fitting and Monte Carlo sampling of fitted parameters, 1 for Monte Carlo sampling of input parameter
    :param distribution:different types of distributions, 0/weibull distribution, 1/normal distribution, 2/gamma distribution
    :param function: the form of the function, default is func1(exponential degradation model), user-defined function must take first parameter as independent variable
    :param m:the number of the increased data set
    :param p0:initial value of parameters
    :param output_file: type to save file,0:mat,1:xlsx,2:npy,3:csv,4:txt
    :param output_image: type to save image,0:png,1:jpg,2:svg,3:pdf
    :return:
    '''

    if m < 1:
        abort(400, "ERROR: The number of augmentations must be a positive integer.")
    if m > 100000:
        abort(400, "ERROR: The number of the increased data set is excessive.")
    else:
        pass

    if function_select == 0:
        function = func1
        p0 = (a, b)
    elif function_select == 1:
        function = func2
        p0 = (a, b, c)
    elif function_select == 2:
        function = func3
        p0 = (a, b, c, d, e)
    elif function_select == 3:
        function = func4
        # p0 = (a, b, c)
        p0 = None
    elif function_select == 4:
        function = func5
        # p0 = (a, b, c, d)
        p0 = None
    elif function_select == 5:
        function = func6
        p0 = (a, b, c, d)
        # p0 = None

    if mode == 0:
        data = []
        data1, _ = Monte_writein(loadpath)
        data.append(data1)
        files = os.listdir(loadpath)
        data = []
        for file in files:
            data1, error = Monte_writein(loadpath + "/" + file)
            if error == 0:
                data.append(data1)
        # plot original degration trajectories
        plt.figure(figsize=(8, 5))
        for i in range(len(data)):
            HI = data[i]
            plt.plot(HI, linewidth=0.5)
        if output_image == 0:
            file_name = "%s.png" % "Original_degradation_trajectories"
        elif output_image == 1:
            file_name = "%s.jpg" % "Original_degradation_trajectories"
        elif output_image == 2:
            file_name = "%s.svg" % "Original_degradation_trajectories"
        elif output_image == 3:
            file_name = "%s.pdf" % "Original_degradation_trajectories"
        plt.title('Original degradation trajectories', fontsize=16)
        plt.xlabel('Time [min]')
        plt.ylabel('Health indicator')
        path = os.path.join(save_path, file_name)
        plt.savefig(path)
        # plt.show()
        plt.close()

        parameter, thresh = func_fit(data, function, p0, save_path)

        if distribution == 2:  # 针对Gamma，删除负参数
            i = 0
            index = []  # Gamma

            while i < len(parameter):
                if function_select == 1:
                    # condition = np.any(parameter[i, :] < 0)
                    condition = np.any(parameter[i, :] < 0) & np.any(parameter[i, :] > -0.00000001)
                else:
                    condition = np.any(parameter[i, :] < 0)
                if condition:
                    index.append(i)
                    parameter = np.delete(parameter, index, axis=0)
                i = i + 1

            min = np.absolute(np.min(parameter)) + 0.0000001
            for i in range(len(parameter)):
                for a in range(parameter.shape[1]):
                    if parameter[i, a] <= 0:
                        parameter[i, a] = parameter[i, a] + min

        new_parameter = np.zeros([m, function.__code__.co_argcount - 1])
        for i in range(function.__code__.co_argcount - 1):
            new_parameter[:, i] = monte_carlo(parameter[:, i], distribution, m)

        # 保存新的参数
        save_data_func(data=new_parameter, output_file=output_file, save_path=save_path,
                       file_name="Monte_carlo_parameters", index_label="Parameters")

        # plot new trajectory with default function and calculated threshhold
        new_data = []
        i0 = -1
        for i in range(m):
            # new_time = np.arange(int(new_time / new_parameter[i][1]))
            if function_select == 0:  # distribution=0和1，2都跑的通
                x = sp.Symbol('x')
                i0 = i0 + 1
                new_time = sp.solve(new_parameter[i][0] * sp.exp(x) - 1 - thresh, x)
                # print(new_parameter[i])
                if isinstance(new_time[0], sp.core.add.Add) == True:
                    new_time[0] = complex(new_time[0])
                    new_time[0] = new_time[0].real

                new_time = np.arange((new_time / new_parameter[i][1]).astype(np.int))

                new_data.append(func1(new_time, *new_parameter[i]))
            elif function_select == 1:  # distribution=0和1都跑的通
                x = sp.Symbol('x')
                i0 = i0 + 1
                new_time = sp.solve(new_parameter[i][0] * sp.exp(x) + new_parameter[i][2] - thresh, x)

                if isinstance(new_time[0], sp.core.add.Add) == True:
                    new_time[0] = complex(new_time[0])
                    new_time[0] = new_time[0].real

                if (new_time / new_parameter[i][1]) < 0:  # 将t小于0的值舍去
                    i0 = i0 - 1
                    # print("io:",i0)
                    continue

                new_time = np.arange((new_time / new_parameter[i][1]).astype(np.int))

                new_data.append(func2(new_time, *new_parameter[i]))
            elif function_select == 2:
                i0 = i0 + 1
                x = sp.Symbol('x', real=True, imaginary=False, positive=True)
                new_time = sp.nsolve(
                    new_parameter[i][0] * sp.exp(x * (new_parameter[i][1])) + new_parameter[i][2] * sp.exp(
                        x * new_parameter[i][3]) + new_parameter[i][4] - thresh, x, (0, 10000),
                    solver='bisect', verify=False)

                new_time = float(new_time)
                new_time = np.arange(int(new_time))
                new_data.append(func3(new_time, *new_parameter[i]))
            elif function_select == 3:  # distribution=0和1都跑通了
                x = sp.Symbol('x')
                new_time = sp.solve(
                    new_parameter[i][0] * np.power(x, 2) + new_parameter[i][1] * x + new_parameter[i][2] - thresh, x)
                if isinstance(new_time[1], sp.core.add.Add) == True:
                    # print(True)
                    new_time[1] = complex(new_time[1])
                    new_time[1] = new_time[1].real

                i0 = i0 + 1
                if new_time[1] < 0:  # 将t小于0的值舍去
                    i0 = i0 - 1
                    # print("io:",i0)
                    continue

                new_time = np.arange(int(new_time[1]))
                new_data.append(func4(new_time, *new_parameter[i]))

            elif function_select == 4:  # Gamma的拟合图阶梯下降
                x = sp.Symbol('x')
                i0 = i0 + 1
                new_time = sp.solve(
                    new_parameter[i][0] * np.power(x, 3) + new_parameter[i][1] * np.power(x, 2) + new_parameter[i][
                        2] * x * new_parameter[i][3] - thresh, x)

                for j in range(len(new_time)):
                    if isinstance(new_time[j], sp.core.add.Add) == True:
                        new_time[j] = complex(new_time[j])
                        new_time[j] = new_time[j].real
                new_time = max(new_time)

                if new_time < 0:  # 将t小于0的值舍去
                    i0 = i0 - 1
                    # print("io:",i0)
                    continue
                new_time = np.arange(int(new_time))
                new_data.append(func5(new_time, *new_parameter[i]))
            elif function_select == 5:  # distribution=0和1都跑不通
                x = sp.Symbol('x')
                i0 = i0 + 1
                new_time = sp.nsolve(
                    new_parameter[i][0] * sp.exp(new_parameter[i][1] * x) + new_parameter[i][2] * np.power(x, 2) +
                    new_parameter[i][3] - thresh, x, (0, 10000), solver='bisect', verify=False)

                if new_time < 0:  # 将t小于0的值舍去
                    i0 = i0 - 1
                    # print("io:",i0)
                    continue
                new_time = np.arange(int(new_time))
                new_data.append(func6(new_time, *new_parameter[i]))

            plt.plot(new_data[i0], linewidth=0.5)
            plt.xlim((0, 2500))
            plt.title('Degradation trajectories after Monte Carlo sampling', fontsize=16)
            plt.xlabel('Time/min')
            plt.ylabel('Health Indicator')

        # 保存剧增后数据
        save_data_func(data=new_data, output_file=output_file, save_path=save_path,
                       file_name="Monte_carlo_data", index_label="Augmentation signal")

        if output_image == 0:
            file_name = "%s.png" % "Degradation_trajectories_after_Monte_Carlo_sampling"
        elif output_image == 1:
            file_name = "%s.jpg" % "Degradation_trajectories_after_Monte_Carlo_sampling"
        elif output_image == 2:
            file_name = "%s.svg" % "Degradation_trajectories_after_Monte_Carlo_sampling"
        elif output_image == 3:
            file_name = "%s.pdf" % "Degradation_trajectories_after_Monte_Carlo_sampling"
        path = os.path.join(save_path, file_name)
        plt.savefig(path)
        # plt.show()
        plt.close()

        # return new_parameter,thresh
        return new_data
    if mode == 1:
        # mat = io.loadmat(loadpath)
        # keys = list(mat.keys())
        # data = mat[keys[3]]
        data = writein(loadpath, 1)
        new_data = monte_carlo(data, distribution, m)
        # 保存新参数
        save_data_func(data=new_data, output_file=output_file, save_path=save_path,
                       file_name="Monte_carlo_parameters", index_label="Parameters")

        return new_data


if __name__ == "__main__":
    # mode 0,  folder "24" contains 24 .mat files of 24 trajectories
    new_data = Monte_Carlo_DA('./folder', mode=0)
    # mode 1, "theta.mat" contains data of the parameter theta that needed to be increased
    # new_parameter2 = Monte_Carlo_DA('theta.mat', mode=1)



