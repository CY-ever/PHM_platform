#!/usr/bin/env python
# coding: utf-8

""" SA optimizer """
import os

import numpy as np
import math
from matplotlib import pyplot as plt

from moduls.ml.dataset import import_data
from moduls.ml.utils import print_params


class SA:

    def __init__(self, objective, initial_temp, final_temp, alpha, max_iter,
                 var_size, x_train, x_test, y_train, y_test,
                 rul_pre=False, candidate=None, net="None",
                 output_image=0, save_path=None):
        """
        :param objective: cost function as an objective
        :param initial_temp: double, manually set initial_temp, e.g. 500
        :param final_temp: double, stop_temp, e.g. 1
        :param alpha: double, temperature changing step, normal range[0.900, 0.999], e.g.0.9
        :param max_iter: int, maximal iteration number e.g. 30
        :param var_size: list, upper and lower bounds of each parameter
        :param net: choose between "DBN", "CNN", "SVM"
        """
        self.objective = objective  # Objective network to be optimize
        # self.path_to_data = path_to_data
        self.initial_temp = initial_temp  # 200
        self.final_temp = final_temp  # 10
        self.alpha = alpha  # 0.9 衰减因子
        self.max_iter = max_iter  # maximal iteration number
        self.var_size = var_size  # [[],[],[]]
        self.dim = np.zeros(len(var_size))
        self.net = net
        self.candidate = candidate
        self.temp = []
        self.states = []
        self.costs = []
        self.current_temp = self.initial_temp  # initialisation of temp
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.save_path = save_path
        self.params_dict = {}   # 优化后得参数结果
        self.rul_pre = rul_pre
        self.output_image = output_image

    def run(self):
        """
        Optimize object network with the simulated annealing algorithm.

        :outcome: cost of all iteration and objective functions' optimized parameters
        """

        state = self._random_start()  # start from a random state, multiple dimension
        self.temp = [self.current_temp]
        if self.net in ["AE", ]:
            cost = self.objective(self.x_train, self.x_test, self.y_train, self.y_test, state,
                                  rul_pre=self.rul_pre,
                                  savePath=None)[1]
            if cost <= 0:
                cost = 0
        elif self.net in ["KNN", "DBN", "ET"]:
            cost = self.objective(state, self.rul_pre)[1]
        elif self.net in ["CNN",]:
            cost = self.objective(state)
            if cost <= 0:
                cost = 0
        elif self.net in ["LSTM", "RF", "DT"]:
            cost = self.objective(self.x_train, self.x_test, self.y_train, self.y_test, state, rul_pre=self.rul_pre)[
                1]
            if cost <= 0:
                cost = 0
            cost = 1 - cost
        else:
            cost = self.objective(state)
            if cost <= 0:
                cost = 0
            cost = 1 - cost

        self.states, self.costs = [state], [cost]
        num_itr = 1
        while self.current_temp > self.final_temp:
            print("====iteration====", num_itr, "...")
            old_state = state
            print("state0", state)
            new_state = self._random_neighbour(old_state)
            print("state1", state)
            print("new_state", new_state)
            # Check if neighbor is best so far

            if self.net in ["AE", ]:
                new_cost = self.objective(self.x_train, self.x_test, self.y_train, self.y_test, new_state,
                                          rul_pre=self.rul_pre, savePath=None)[1]
                if cost <= 0:
                    cost = 0
            elif self.net in ["KNN", "DBN", "ET"]:
                new_cost = self.objective(new_state, self.rul_pre)[1]
            elif self.net in ["CNN", ]:
                new_cost = self.objective(new_state)
                if new_cost <= 0:
                    new_cost = 0
            elif self.net in ["LSTM", "RF", "DT"]:
                new_cost = \
                self.objective(self.x_train, self.x_test, self.y_train, self.y_test, new_state, rul_pre=self.rul_pre)[
                    1]
                if new_cost <= 0:
                    new_cost = 0
                new_cost = 1 - new_cost
            else:
                new_cost = self.objective(new_state)
                if new_cost <= 0:
                    new_cost = 0
            # new_cost = self.objective(new_state)

            cost_diff = new_cost - cost
            print("cost", cost, "new_cost", new_cost)
            print("cost_diff", cost_diff)
            # if the new solution is better, accept it
            if cost_diff < 0:
                state = new_state
                cost = new_cost
                print("==>accept new")
                # if the new solution is not better, accept it with a probability of e^(-cost/temp)
            elif cost_diff >= 0:
                if np.random.uniform(0, 1) < math.exp(-(cost_diff*1.38064852*10**23) / self.current_temp):
                    state = new_state
                    cost = new_cost
                    print("==>accept new")
                else:
                    print("==>reject new")
            self.states.append(state)
            self.costs.append(cost)
            self.temp.append(self.current_temp)
            print("solution", state)
            print("cost", cost)
            print("T", self.current_temp)
            # reduce the temperature
            self.current_temp = self.current_temp*self.alpha
            num_itr += 1
            self.params_dict = print_params(state, self.candidate, net=self.net)
            print(len(self.temp), len(self.costs))
            if num_itr >= self.max_iter:
                print("reach the maximal iteration number")
                break
        self.plot_curve()
        return self.params_dict

    def plot_curve(self):
        """
        Plot optimizer curve with iteration

        :return: None
        """
        x = np.arange(1, len(self.costs)+1)
        y = self.costs
        plt.plot(x, y)
        # plt.plot(y)

        plt.xticks(x)
        # plt.plot( self.costs)
        plt.ylabel("Objective costs")
        plt.xlabel("Iteration")
        plt.title("Optimization curve")

        # file_name = self.net + "_SA.png"
        # file_name = "optimization_curve.png"
        if self.save_path:
            if self.output_image == 1:
                plt.savefig(os.path.join(self.save_path, "optimization_curve.jpg"))
            elif self.output_image == 2:
                plt.savefig(os.path.join(self.save_path, "optimization_curve.svg"))
            elif self.output_image == 3:
                plt.savefig(os.path.join(self.save_path, "optimization_curve.pdf"))
            else:
                plt.savefig(os.path.join(self.save_path, "optimization_curve.png"))
        else:
            plt.show()

        plt.close()

    def _random_start(self):
        """
        Random start point in the given interval

        :return: random state value
        """
        print("___START____")
        # rd_state = np.zeros(len(self.var_size))

        # for i in range(len(np.zeros(len(self.var_size)))):
        #     rd_point = np.random.uniform(self.var_size[i][0], self.var_size[i][1])
        #     rd_state[i] = rd_point

        # for i in range(len(self.var_size)):
        #     if self.var_size[i][0] == self.var_size[i][1]:
        #         rd_point = self.var_size[i][0]
        #         rd_state[i] = rd_point
        #     elif isinstance(self.var_size[i][0], int):
        #         rd_point = np.random.randint(low=self.var_size[i][0],
        #                                      high=self.var_size[i][1])
        #         rd_state[i] = rd_point
        #     elif isinstance(self.var_size[i][0], float):
        #         rd_point = np.random.uniform(low=self.var_size[i][0],
        #                                      high=self.var_size[i][1])
        #         rd_state[i] = rd_point
        #     else:
        #         rd_point = np.random.choice(self.var_size[i])
        #         rd_state[i] = rd_point
        rd_state = []
        for g in range(len(self.var_size)):
            if self.var_size[g][0] == self.var_size[g][1]:
                rd_state.append(self.var_size[g][0])
            elif isinstance(self.var_size[g][0], int):
                rd_state.append(np.random.randint(low=self.var_size[g][0],
                                                  high=self.var_size[g][1]))
            elif isinstance(self.var_size[g][0], float):
                rd_state.append(np.random.uniform(low=self.var_size[g][0],
                                                  high=self.var_size[g][1]))
            else:
                rd_state.append(np.random.choice(self.var_size[g]))

        print("init_random_state", rd_state)
        return rd_state

    def _random_neighbour(self, state_old):
        """
        Find neighbour of current state

        :param state_old: list, old state
        """
        print("___NEIGHBOUR____")

        # neighbour = np.zeros(len(self.var_size))
        neighbour = []
        for j in range(len(np.zeros(len(self.var_size)))):
            if isinstance(self.var_size[j][0], str):
                neighbour.append(self.var_size[j][np.random.randint(low=0, high=len(self.var_size[j]))])
                # neighbour[j] = self.var_size[j][np.random.randint(low=0, high=len(self.var_size[j]))]
            elif isinstance(self.var_size[j][0], int):
                amplitude = (self.var_size[j][1] - self.var_size[j][0]) * 1 / 10
                delta = (-amplitude / 2.) + amplitude * np.random.random_sample()
                middle_point = state_old[j]
                neighbour_point = max(min(middle_point + delta, self.var_size[j][1]), self.var_size[j][0])
                # neighbour[j] = neighbour_point
                neighbour.append(int(neighbour_point))
            else:
                amplitude = (self.var_size[j][1] - self.var_size[j][0]) * 1 / 10
                delta = (-amplitude / 2.) + amplitude * np.random.random_sample()
                middle_point = state_old[j]
                neighbour_point = max(min(middle_point + delta, self.var_size[j][1]), self.var_size[j][0])
                # neighbour[j] = neighbour_point
                neighbour.append(neighbour_point)
        return neighbour
