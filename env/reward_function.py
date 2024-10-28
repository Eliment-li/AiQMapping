import math

import utils.common_utils as common_utils
import core.chip  as chip
class RewardFunction:

    '''
    compute_total_distance
    '''
    def rfv1(self,env, action):
        reward = env.stop_thresh
        # 计算距离
        distance = common_utils.compute_total_distance(env.position)
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
        distance = common_utils.compute_total_distance(env.position)
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


    def rfv3(self,env, action):
        reward = env.stop_thresh
        terminated =False
        # 计算距离
        distance = common_utils.compute_total_distance(env.position)
        #cu.swap_counts(circuit_name=env.circuit,initial_layout=env.occupy)

        d1 = (env.default_distance - distance) / env.default_distance
        d2 = (env.last_distance - distance) / env.last_distance
        env.last_distance = distance

         # 计算是否满足连接性
        cnt =  chip.cnt_meet_nn_constrain(env.nn, env.occupy)
        n1 = (env.default_nn - cnt) / (env.default_nn + 1)
        n2 = (env.last_nn - cnt) / (env.last_nn + 1)


        k1 = d1 + n1
        k2 = d2 + n2

        if k1 == 0:
            k1 = 0.5
        if k2 > 0:
            reward = (math.pow((1 + k2), 2) - 1) * math.fabs(k1)
        elif k2 < 0:
            reward = -1 * (math.pow((1 - k2), 2) - 1) * math.fabs(k1)
        else:
            reward = 0

        if cnt > env.max_nn_meet:
            reward +=  cnt
            env.max_nn_meet = cnt
        if cnt == len(env.nn):
            reward  =  2 * cnt
            terminated =True
        if reward == 0:
            reward = -0.04

        return reward,terminated


