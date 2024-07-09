# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 15:29:08 2017
@author: www
"""

# 本程序是用四阶龙格库塔法求解课本（数值计算方法 马东升）P242页的例7-3
# fun为指定的导数的函数
# rf4为四阶龙格库塔法

import numpy as np
from moduls.bearing_simulation.bearing_simulation_with_defects.bearing_dynamics_model import bearing_dynamics_model
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def fun(t, x):
    # dx = x - (2 * t / x)
    dx = np.cos(x)
    return dx


def rf(tspan, y0):

    ys = odeint(fun, y0, tspan)
    ys = np.array(ys).flatten()
    plt.figure(dpi=800)
    plt.plot(tspan, ys)
    plt.show()
    plt.close()
    return tspan, ys


# input
#     t,y0:初始给出的t值，y0值
#     h   ：步长
#     N   ：迭代次数
# print
#    x1,y1:每次迭代输出的结果
# def rf4(t, y0, h, N):
#     t_all = []
#     y_all = []
#     n = 1
#     while n != N:
#         t1 = t + h
#         k1 = fun(t, y0)
#         k2 = fun(t + h / 2, y0 + h * k1 / 2)
#         k3 = fun(t + h / 2, y0 + h * k2 / 2)
#         k4 = fun(t1, y0 + h * k3)
#         # k1 = fun(t, y0)
#         # k2 = fun(t + h / 2, y0 + h * k1 / 2)
#         # k3 = fun(t + h / 2, y0 + h * k2 / 2)
#         # k4 = fun(t1, y0 + h * k3)
#         y1 = y0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
#         # print(t1, y1)
#         t_all.append(t1)
#         y_all.append(y1)
#         n = n + 1
#         t = t1
#         y0 = y1
#
#     plt.figure()
#     plt.plot(t_all, y_all)
#     plt.show()
#
#     return t_all, y_all


def main():
    # time, y = rf4(0, 0, 0.01, 1000)
    # t = np.linspace(0, 5, 100)
    t = np.arange(0, 20, step=0.001)
    time, y = rf(t, 0)

    print(time)
    print(y)


if __name__ == '__main__':
    main()
