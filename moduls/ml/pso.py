"""Particle Swarm Optimization"""
import os

import numpy as np
from matplotlib import pyplot as plt

from moduls.ml.dataset import import_data
from moduls.ml.utils import print_params


class Particle:
    def __init__(self, Pos=None, Vel=None, Cost=None, Best_pos=None, Best_cost=None):
        """
        Structure of a particle.

        :param Pos: list, parameters of this particle
        :param Vel: list, searching speed of this particle
        :param Cost: float, current cost of this particle
        :param Best_pos: list, during whole optimization the best parameters
        :param Best_cost: float, best cost in the searching history
        """
        self.Pos = Pos
        self.Vel = Vel
        self.Cost = Cost
        self.Best_pos = Best_pos
        self.Best_cost = Best_cost


class PSO:
    def __init__(self, objective, part_num, num_itr, var_size,
                 x_train, x_test, y_train, y_test, rul_pre=False,
                 candidate=None, net=None, output_image=0, save_path=None):
        """
        Particle Swarm Optimization
        :param objective: cost function as an objective
        :param part_num: integer, number of particles
        :param num_itr: integer, number of iterations
        :param var_size: list, 参数上下限,upper and lower bounds of each parameter,
                        as in [[x1_min,x1_max], [x2_min,x2_max],..., [xn_min,xn_max]]
        :param candidate: list, candidates of the discrete parameters
        :param net: string, name of the optimized network
        """
        self.part_num = part_num  # Number of the particles
        self.dim = len(var_size)  # Dimension of the particle
        self.num_itr = num_itr  # Run how many iterations
        self.objective = objective  # Objective function to be optimize
        self.rul_pre = rul_pre
        # self.path_to_data = path_to_data
        self.w = 0.9  # initial weight
        self.c1 = 1.49
        self.c2 = 1.49
        self.var_size = var_size  # Length must correspond to the dimension of particle
        self.vmax = 1  # Maximum search velocity
        self.vmin = 0.01  # Minimum search velocity
        self.GlobalBest_Cost = 1e5
        self.GlobalBest_Pos = []
        # Array to hold Best costs on each iterations
        self.Best_Cost = []
        # Save space for particles
        self.particle = []
        assert self.dim == len(self.var_size)
        self.net = net
        # self.x_train, self.x_test, self.y_train, self.y_test = import_data(self.path_to_data, model=self.net)
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
        self.candidate = candidate
        self.save_path = save_path
        self.params_dict = {}
        self.output_image = output_image

    def init_population(self):
        """
        Initialize all the particles and find the temporary best parameter.

        :return: None
        """
        print('Initializing...')
        for i in range(self.part_num):
            x = Particle()
            # initialize random position

            # x.Pos = np.zeros(self.dim)
            # for j in range(len(x.Pos)):
            #     x.Pos[j] = np.random.uniform(self.var_size[j][0], self.var_size[j][1])

            x.Pos = []
            for g in range(len(self.var_size)):
                if self.var_size[g][0] == self.var_size[g][1]:
                    x.Pos.append(self.var_size[g][0])
                elif isinstance(self.var_size[g][0], int):
                    x.Pos.append(np.random.randint(low=self.var_size[g][0],
                                                   high=self.var_size[g][1]))
                elif isinstance(self.var_size[g][0], float):
                    x.Pos.append(np.random.uniform(low=self.var_size[g][0],
                                                   high=self.var_size[g][1]))
                else:
                    x.Pos.append(np.random.choice(self.var_size[g]))

            # calculate cost from random parameters
            # print(x.Pos)

            # x.Cost = self.objective(x.Pos)
            if self.net in ["AE", ]:
                x.Cost = self.objective(self.x_train, self.x_test, self.y_train, self.y_test, x.Pos, rul_pre=self.rul_pre, savePath=None)[1]
                if x.Cost <= 0:
                    x.Cost = 0
            elif self.net in ["KNN", "DBN", "ET"]:
                x.Cost = self.objective(x.Pos, rul_pre=self.rul_pre)[1]
                if x.Cost <= 0:
                    x.Cost = 0
            elif self.net in ["LSTM", "RF", "DT"]:
                x.Cost = self.objective(self.x_train, self.x_test, self.y_train, self.y_test, x.Pos, rul_pre=self.rul_pre)[1]
                if x.Cost <= 0:
                    x.Cost = 0
                x.Cost = 1-x.Cost
            elif self.net in ["CNN", ]:
                x.Cost = self.objective(x.Pos)
                if x.Cost <= 0:
                    x.Cost = 0
            else:
                x.Cost = self.objective(x.Pos)
                if x.Cost <= 0:
                    x.Cost = 0

            x.Vel = np.zeros(self.dim)
            x.Best_pos = x.Pos
            x.Best_cost = x.Cost
            self.particle.append(x)

            if self.particle[i].Best_cost < self.GlobalBest_Cost:
                self.GlobalBest_Cost = self.particle[i].Best_cost
                self.GlobalBest_Pos = self.particle[i].Best_pos
        self.Best_Cost.append(self.GlobalBest_Cost)
        print('Initialize complete, with best cost =',
              self.GlobalBest_Cost,
              "\nTemporary best solution:",
              self.GlobalBest_Pos)

    def iterator(self):
        """
        Run the iterations to find the best parameters.

        :return: None
        """
        print('Iterator running...')
        for i in range(self.num_itr):
            for j in range(self.part_num):
                # create r1,r2
                r1 = np.random.uniform(self.vmin, self.vmax, self.dim)
                r2 = np.random.uniform(self.vmin, self.vmax, self.dim)
                # Update
                # self.particle[j].Vel = self.w * self.particle[j].Vel \
                #                        + self.c1 * r1 * (self.particle[j].Best_pos - self.particle[j].Pos) \
                #                        + self.c2 * r2 * (self.GlobalBest_Pos - self.particle[j].Pos)
                # self.particle[j].Pos = self.particle[j].Pos + self.particle[j].Vel
                # Update
                self.particle[j].Vel = list(self.particle[j].Vel)
                for h in range(len(self.particle[j].Pos)):
                    if isinstance(self.particle[j].Pos[h], str):

                        self.particle[j].Vel[h] = self.particle[j].Pos[h]
                        self.particle[j].Pos[h] = self.particle[j].Vel[h]
                    elif isinstance(self.particle[j].Pos[h], int):

                        self.particle[j].Vel[h] = self.w * self.particle[j].Vel[h] \
                                                  + self.c1 * r1[h] * (
                                                              self.particle[j].Best_pos[h] - self.particle[j].Pos[h]) \
                                                  + self.c2 * r2[h] * (self.GlobalBest_Pos[h] - self.particle[j].Pos[h])
                        self.particle[j].Pos[h] = int(self.particle[j].Pos[h] + self.particle[j].Vel[h])
                    else:
                        self.particle[j].Vel[h] = self.w * self.particle[j].Vel[h] \
                                                  + self.c1 * r1[h] * (
                                                              self.particle[j].Best_pos[h] - self.particle[j].Pos[h]) \
                                                  + self.c2 * r2[h] * (self.GlobalBest_Pos[h] - self.particle[j].Pos[h])
                        self.particle[j].Pos[h] = self.particle[j].Pos[h] + self.particle[j].Vel[h]

                # Check whether position out of search space
                for x in range(len(self.particle[j].Pos)):
                    if isinstance(self.particle[j].Pos[x], str):
                        continue
                    elif self.particle[j].Pos[x] > self.var_size[x][1]:
                        self.particle[j].Pos[x] = self.var_size[x][1]
                    elif self.particle[j].Pos[x] < self.var_size[x][0]:
                        self.particle[j].Pos[x] = self.var_size[x][0]
                    assert self.var_size[x][1] >= self.particle[j].Pos[x] >= self.var_size[x][0]
                # self.particle[j].Pos[2] = int(self.particle[j].Pos[2])
                # Recalculate cost
                # print(self.particle[j].Pos)

                # self.particle[j].Cost = self.objective(self.particle[j].Pos)
                if self.net in ["AE", ]:
                    self.particle[j].Cost = \
                    self.objective(self.x_train, self.x_test, self.y_train, self.y_test, self.particle[j].Pos,
                                   rul_pre=self.rul_pre, savePath=None)[1]
                    if self.particle[j].Cost <= 0:
                        self.particle[j].Cost = 0
                elif self.net in ["KNN", "DBN", "ET"]:
                    self.particle[j].Cost = self.objective(self.particle[j].Pos, rul_pre=self.rul_pre)[1]
                    if self.particle[j].Cost <= 0:
                        self.particle[j].Cost = 0
                elif self.net in ["LSTM", "RF", "DT"]:
                    self.particle[j].Cost = \
                    self.objective(self.x_train, self.x_test, self.y_train, self.y_test, self.particle[j].Pos, rul_pre=self.rul_pre)[1]
                    if self.particle[j].Cost <= 0:
                        self.particle[j].Cost = 0
                    self.particle[j].Cost = 1 - self.particle[j].Cost
                elif self.net in ["CNN", ]:
                    self.particle[j].Cost = self.objective(self.particle[j].Pos)
                    if self.particle[j].Cost <= 0:
                        self.particle[j].Cost = 0
                else:
                    self.particle[j].Cost = self.objective(self.particle[j].Pos)
                    if self.particle[j].Cost <= 0:
                        self.particle[j].Cost = 0

                print("Current cost=", self.particle[j].Cost, "With position:", self.particle[j].Pos)
                if self.particle[j].Cost < self.particle[j].Best_cost:
                    self.particle[j].Best_cost = self.particle[j].Cost
                    self.particle[j].Best_pos = self.particle[j].Pos
                    print("Find better personel best, Updating with pos:", self.particle[j].Pos)
                    if self.particle[j].Best_cost < self.GlobalBest_Cost:
                        self.GlobalBest_Cost = self.particle[j].Best_cost
                        self.GlobalBest_Pos = self.particle[j].Best_pos
                        print("Find better global solution, Updating with pos:", self.particle[j].Pos)
                    else:
                        print("Not better than previous global solution, dropping...")
                else:
                    print("Not better than previous personal best, dropping...")
            self.Best_Cost.append(self.GlobalBest_Cost)
            self.w = self.w * 0.9
            print()
            print('iteration', i + 1, ': Cost=', self.GlobalBest_Cost)
            print_params(self.GlobalBest_Pos, self.candidate, net=self.net)

    def plot_curve(self):
        """
        Plot optimizer curve

        :return: None
        """
        # plt.plot(self.Best_Cost)

        x = np.arange(1, len(self.Best_Cost) + 1)
        y = self.Best_Cost
        plt.plot(x, y)

        plt.xticks(x)
        plt.ylabel("Objective costs")
        plt.xlabel("Iteration")
        plt.title("Optimization curve")

        # file_name = self.net + "_PSO.png"
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

    def run(self):
        """
        General call for the whole optimization.

        :return: None
        """
        print('PSO start running...')
        self.init_population()
        self.iterator()
        print("Iteration completed.")
        self.plot_curve()
        self.params_dict = print_params(self.GlobalBest_Pos, self.candidate, net=self.net)
        return self.params_dict
