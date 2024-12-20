import math

import numpy as np
import torch

import utils.common_utils as comu
import  utils.circuits_util as cu
from core import chip
from scipy.special import expit

class RewardFunction:

    '''
    compute_total_distance
    '''
    def rfv1(self,env, action):
        reward = env.stop_thresh
        # 计算距离
        distance = comu.compute_total_distance(env.position)
        #cu.swap_counts(circuit_name=env.circuit,initial_layout=env.occupy)

        d1 = (env.default_distance - distance) / env.default_distance
        d2 = (env.last_distance - distance) / env.last_distance
        env.last_distance = distance

        error = 0
        for v in env.occupy:
            error += chip.QUBITS_ERROR_RATE[v]

        e1 = (env.default_error - error) / env.default_error
        e2 = (env.last_error - error) / env.last_error
        env.last_error = error

        k1 = 0.8 * d1 + 0.2 * e1
        k2 = 0.8 * d2 + 0.2 * e2

        if k1 == 0:
            k1 = 0.5
        if k2 > 0:
            reward = (math.pow((1 + k2), 2) - 1) * math.fabs(k1)
        elif k2 < 0:
            reward = -1 * (math.pow((1 - k2), 2) - 1) * math.fabs(k1)
        else:
            reward = 0
        #reward *=1.5
        return reward

    def rfv2(self,env, action):
        reward = env.stop_thresh
        # 计算距离
        distance = comu.compute_total_distance(env.position)
        #cu.swap_counts(circuit_name=env.circuit,initial_layout=env.occupy)

        d1 = (env.default_distance - distance) / env.default_distance
        d2 = (env.last_distance - distance) / env.last_distance
        env.last_distance = distance

        k1 = d1
        k2 = d2

        if k1 == 0:
            k1 = 0.5
        if k2 > 0:
            reward = (math.pow((1 + k2), 2) - 1) * math.fabs(k1)
        elif k2 < 0:
            reward = -1 * (math.pow((1 - k2), 2) - 1) * math.fabs(k1)
        else:
            reward = 0
        #reward *=1.5
        return reward

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    #可基于Qiskit达到训练目标
    def rfv3(self,env, action):
        reward = env.stop_thresh
        terminated =False
        # 计算距离
        #distance = comu.compute_total_distance(env.position)
        distance = cu.swap_counts(circuit=env.QiskitCircuit,initial_layout=env.occupy)

        k1 = (env.default_distance - distance) / env.default_distance
        k2 = (env.last_distance - distance) / env.last_distance
        env.last_distance = distance

        if distance == 0:
            return 4, True

        if k2 > 0:
            #Expit (a.k.a. logistic sigmoid) ufunc for ndarrays.
            reward = (math.pow((1 + k2), 2) - 1) * (expit(1 + k1))
        elif k2 < 0:
            reward = -2 * (math.pow((1 - k2), 2) - 1) * (expit(1 + k1))
            if distance - env.last_distance  <= 1:
                reward *= 1.25
        else:
            reward = -0.05

        return reward,terminated


    #基于distance 计算快
    def rfv4(self, env, action):

        reward = env.stop_thresh
        terminated = False
        # 计算距离
        distance = comu.compute_total_distance(env.position)

        k1 = (env.default_distance - distance) / env.default_distance
        k2 = (env.last_distance - distance) / env.last_distance
        env.last_distance = distance

        if k2 > 0:
            r1 = (math.pow((1 + k2), 2) - 1) * (1 + np.tanh(k1))
        elif k2 < 0:
            r1= -2.5 * (math.pow((1 - k2), 2) - 1) * (1 - np.tanh(k1))
        else:
            r1 = -0.1

        # 计算是否满足连接性
        nn = chip.cnt_meet_nn_constrain(env.nn, env.occupy)
        n1 = (nn - env.default_nn) / (env.default_nn + 1)
        n2 = (nn - env.last_nn) / (env.last_nn + 1)
        env.last_nn = nn

        if n2 > 0:
            r2 = (math.pow((1 + n2), 2) - 1) * (1 + np.tanh(n1))
        elif n2 < 0:
            r2= -1 * (math.pow((1 - n2), 2) - 1) * (1 - np.tanh(n1))
        else:
            r2 = -0.1

        #完全满足
        if nn == len(env.nn):
            terminated =True

        reward = r1

        return reward, terminated

    #基于新的精确 distance
    def rfv5(self, env, action):

        reward = env.stop_thresh
        terminated = False
        # 计算距离
        distance = chip.chip_Qubit_distance(env.nn, env.occupy)

        k1 = (env.default_distance - distance) / env.default_distance
        k2 = (env.last_distance - distance) / env.last_distance


        if k2 > 0:
            r1 = (math.pow((1 + k2), 2) - 1) * (1 + np.tanh(k1))
        elif k2 < 0:
            #后缀 -0.1, 防止 agent 利用漏洞, see: 11-21 lr_schedule xeb5-9.md 总结部分
            r1 = -3.5 * (math.pow((1 - k2), 2) - 1) * (1 - np.tanh(k1))
            if distance - env.last_distance  <=1:
                r1 *= 1.25
        else:
            r1 = -0.05

        env.last_distance = distance
        reward = r1

        return reward, terminated

    #加上 qubite error rate
    def rfv6(self,env,action):
        error_rate = 0
        for Q in env.occupy:
            error_rate += chip.QUBITS_ERROR_RATE_MAP[Q]
        pass





