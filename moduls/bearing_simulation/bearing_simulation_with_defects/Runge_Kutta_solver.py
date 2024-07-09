# Generated with SMOP  0.41

# Runge_Kutta_solver.m

## the code was developed to call the "ode45" solution method in matlab to solve "bearing_dynamics_model".
# author: Diwang Ruan
# date: 2020.09.09
# version: V1.0

## introduction of "Runge_Kutta_solver"
# the main function is to solve the "bearing_dynamics_model" with embedded ode45
# inputs are 8 structures defined in "bearing_simulation_with_defects"
# outputs are time, acceleration in x-axis(acc_x) and y-axis(acc_y), units: s, m/s^2,  m/s^2;
# in the end, it will call "result_analysis_plot" to present simulation results

## the definition of  "Runge_Kutta_solver"

import numpy as np
from moduls.bearing_simulation.bearing_simulation_with_defects.result_analysis_plot import result_analysis_plot
from moduls.bearing_simulation.bearing_simulation_with_defects.excel_generation import excel_generation
from moduls.bearing_simulation.bearing_simulation_with_defects.GlobalVariable import globalVariables as gbv
from moduls.bearing_simulation.bearing_simulation_with_defects.bearing_dynamics_model import bearing_dynamics_model
from scipy.integrate import odeint


# def rf4(t, x, h, N):
#     t_all = []
#     x_all = []
#     n = 1
#     while n != N:
#         t1 = t + h
#         k1 = bearing_dynamics_model(t, x)
#         k2 = bearing_dynamics_model(t + h / 2, x + h * k1 / 2)
#         k3 = bearing_dynamics_model(t + h / 2, x + h * k2 / 2)
#         k4 = bearing_dynamics_model(t1, x + h * k3)
#         x1 = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
#
#         t_all.append(t1)
#         x_all.append(x1)
#         n = n + 1
#         t = t1
#         x = x1
#
#     return t_all, x_all

def rf4(t, x0):

    ys = odeint(bearing_dynamics_model, x0, t)
    # ys = np.array(ys).flatten()
    return t, ys


def Runge_Kutta_solver(bearing_parameter=None, pedestal_parameter=None, shaft_parameter=None, sprung_parameter=None,
                       condition_parameter=None, defect_parameter=None, DoF5_parameter=None, sim_parameter=None,
                       name_parameter=None, save_path="", output_image=0,
                       output_file=0, *args, **kwargs):
    # varargin = Runge_Kutta_solver.varargin
    # nargin = Runge_Kutta_solver.nargin

    # # to assign the input parameters in the workspace for further use


    gbv.bearing_parameter = bearing_parameter
    gbv.pedestal_parameter = pedestal_parameter
    gbv.shaft_parameter = shaft_parameter
    gbv.sprung_parameter = sprung_parameter
    gbv.condition_parameter = condition_parameter
    gbv.defect_parameter = defect_parameter
    gbv.DoF5_parameter = DoF5_parameter
    gbv.sim_parameter = sim_parameter

    # tic
    # to define the necessary simulation configuration parameters

    # tspan = np.arange(0, sim_parameter.sim_duration, sim_parameter.step_size)
    end = sim_parameter.sim_duration
    # num = int(1/sim_parameter.step_size)
    # t = np.linspace(0, end, num)
    t = np.arange(0, end, sim_parameter.step_size)
    # t_size = len(tspan)

    # x_init = np.zeros((10, 1))

    x_init = np.zeros(10)
    # x_init = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # x_init = np.random.rand(10)

    # options = odeset('MaxStep', sim_parameter.max_step, 'RelTol', sim_parameter.absolute_tolerance, 'AbsTol',
    #                  sim_parameter.relative_tolerance)

    # to call the ode45 to solve the bearing dynamics model
    t, x = rf4(t, x_init)
    # t, x = rf4(0, x_init, sim_parameter.step_size, t_size)


    time = t

    # acc_x = x(np.arange(), 2)
    # for i in range(len(x)):
    #     x[i] = list(x[i])

    # new_a = [y for a in a3 for y in a]

    x = np.array(x)

    acc_x = x[:, 1]

    # acc_y = x(np.arange(), 4)
    acc_y = x[:, 3]

    # acc_x=x(:,6);
    # acc_y=x(:,8);
    # toc
    print("开始绘图!")
    # to present the simulation results
    result_analysis_plot(time=time, acc_x=acc_x, acc_y=acc_y, name_parameter=name_parameter,
                         outer_ring_switch=defect_parameter.outer_ring_switch,
                         inner_ring_switch=defect_parameter.inner_ring_switch,
                         ball_switch=defect_parameter.ball_switch,
                         save_path=save_path, output_image=output_image)
    # excel_generation(time, acc_x, output_file=output_file, save_path=save_path)

    return time, acc_x, acc_y


if __name__ == '__main__':
    pass
