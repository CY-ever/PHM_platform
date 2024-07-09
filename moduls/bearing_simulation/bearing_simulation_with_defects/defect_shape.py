# Generated with SMOP  0.41

# defect_shape.m

## the code was designed to model the bearing defect, including defect position, defect size, and defect shape.
# author: Diwang Ruan
# date: 2020.09.09
# version: V1.0

## introduction of "defect_shape"
# the functon was used to model the defect shape with fie different piece-wise function
# input:
# d: ball diameter, m
# L: length of defect, m
# B: width of defect, m
# H: hight of defect, m
# Hd��defect depth at present position, m;
# ball_theta: ball angular position, rad;
# phi_start: the start position of defect, rad;
# delta_phi_d: the span angle of defect, rad;
# output: defect_depth, m��
# the modeling idea comes from paper: "dynamics modeling and analysis of local fault of rolling element bearing"

## defect shape function

import numpy as np


# @function
def defect_shape(d=None, L=None, B=None, H=None, Hd=None, ball_theta=None, phi_start=None, delta_phi_d=None, *args,
                 **kwargs):
    # varargin = defect_shape.varargin
    # nargin = defect_shape.nargin

    # max and min limit definition
    max_limit = 50
    phi_1 = np.dot(delta_phi_d, 0.2)
    phi_2 = np.dot(delta_phi_d, 0.8)
    yita_bd = d / min(L, B)

    yita_d = L / B

    if H < Hd:
        cd_gamma = H
    else:
        cd_gamma = Hd

    # different functions for different defect size
    defect_flag = 0
    # defect_shape.m:40
    # yita_bd 这里是一个几列向量呢？不是单一个数吗？是跟缺陷数量有关？还是跟ball数量有关？
    if all(np.ravel(yita_bd) >= max_limit):
        defect_flag = 1

    if all(np.ravel(yita_bd) > 1) and all(np.ravel(yita_d) <= 1):
        defect_flag = 2

    if all(np.ravel(yita_bd) > 1) and all(np.ravel(yita_d) > 1):
        defect_flag = 3

    if all(np.ravel(yita_bd) <= 1):
        defect_flag = 4

    # function definition for different defect flag
    if 0 == defect_flag:
        cd = 0
    else:
        if 1 == defect_flag:
            H1 = cd_gamma
            cd = H1
        else:
            if 2 == defect_flag:
                phi = ball_theta - phi_start
                H2 = np.dot(cd_gamma, np.sin(np.dot(np.pi / delta_phi_d, phi)))
                cd = H2
            else:
                if 3 == defect_flag:
                    phi = ball_theta - phi_start
                    H3 = 0
                    if phi > 0 and phi < phi_1:
                        H3 = np.dot(cd_gamma, np.sin(np.dot(np.dot(0.5, np.pi) / phi_1, phi)))
                    if phi >= phi_1 and phi < phi_2:
                        H3 = cd_gamma
                    if phi >= phi_2 and phi < delta_phi_d:
                        H3 = np.dot(cd_gamma, np.sin(np.dot(0.5, np.pi) / phi_1 + np.dot(0.5, np.pi)))
                    cd = H3
                else:
                    if 4 == defect_flag:
                        phi = ball_theta - phi_start
                        H4 = 0
                        if phi > 0 and phi < phi_1:
                            H4 = np.dot(cd_gamma, np.sin(np.dot(np.dot(2, np.pi) / phi_1, phi)))
                        if phi >= phi_1 and phi < phi_2:
                            H4 = cd_gamma
                        if phi >= phi_2 and phi < delta_phi_d:
                            H4 = np.dot(cd_gamma, np.sin(np.dot(2, np.pi) / phi_1 + np.dot(0.5, np.pi)))
                        cd = H4
                    else:
                        cd = 0

    defect_depth = cd

    return defect_depth


if __name__ == '__main__':
    pass
