import math

import numpy as np

import utils.common_utils as comu
import  utils.circuits_util as cu
import core.chip  as chip
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
        distance = cu.swap_counts(circuit_name=env.circuit,initial_layout=env.occupy)
        if distance == 0:
            return 4,True

        k1 = (env.default_distance - distance) / env.default_distance
        k2 = (env.last_distance - distance) / env.last_distance
        env.last_distance = distance

        if k2 > 0:
            reward = (math.pow((1 + k2), 2) - 1) * (1 + np.tanh(k1))
        elif k2 < 0:
            reward = -1 * (math.pow((1 - k2), 2) - 1) * (1 - np.tanh(k1))
        else:
            reward = -0.1

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
            r1= -1 * (math.pow((1 - k2), 2) - 1) * (1 - np.tanh(k1))
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




