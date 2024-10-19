import math
import datetime
from copy import copy,deepcopy
from logging import lastResort
from pprint import pprint

import gymnasium as gym
import numpy as np
from gymnasium import spaces, register
from gymnasium.spaces import MultiBinary, MultiDiscrete,Discrete
from gymnasium.spaces.utils import flatten_space
from qiskit_aer import AerSimulator
from loguru import logger
import warnings
from typing import Optional
import os

from core.chip import QUBITS_ERROR_RATE, move_point, grid, COUPLING_SCORE, ADJ_LIST, meet_nn_constrain
from utils.circuits_util import qubits_nn_constrain
from utils.common_utils import compute_total_distance, generate_unique_coordinates, data_normalization, linear_scale
from utils.visualize.trace import show_trace

os.environ["SHARED_MEMORY_USE_LOCK"] = '1'
simulator = AerSimulator()
'''
v3 每次只移动一个对象
记录 trace 以可视化
'''
warnings.filterwarnings("ignore")
class CircuitEnv_v3(gym.Env):
    def __init__(self, config: Optional[dict] = None):
        self.debug = False #config.get('debug')
        self.trace = []
        self.rs = []
        # circuit 变量
        self.circuit = 'XEB_3_qubits_8_cycles_circuit.txt'
        self.qubit_nums = 3

        #chip 变量
        self.position =generate_unique_coordinates(3)
        self.nn = qubits_nn_constrain('XEB_3_qubits_8_cycles_circuit.txt')
        self.grid = copy(grid)

        # 被占据的qubit，用 Q序号为标识
        self.occupy = []
        for p in self.position:
            px = p[0]
            py = p[1]
            self.occupy.append(deepcopy(self.grid[px][py]))
        self.occupy = np.float32(self.occupy)

        self.qubits = np.float32(QUBITS_ERROR_RATE)
        self.coupling= np.float32(COUPLING_SCORE)

        #obs_size = len(self.qubits) + len(self.coupling) + len(self.occupy)
        obs_size = len(self.position)*2
        # todo 先试试 flatten, 后面尝试直接用 spaces.Box
        self.observation_space = flatten_space(spaces.Box(0,1,(1,obs_size),dtype=np.float32,))
        self.obs = linear_scale(np.array(self.position,dtype=np.float32).flatten())
        #self.obs = np.concatenate([self.qubits, self.coupling, linear_scale(self.occupy)],dtype=np.float32)

        self.default_distance = compute_total_distance(self.position)
        self.last_distance = self.default_distance

        #self.action_space = MultiDiscrete(np.array([5] * (self.qubit_nums),dtype=int))
        self.action_space = MultiDiscrete(np.array([self.qubit_nums,5],dtype=int))

        #stop conditions
        self.max_step = 1000
        self.stop_thresh = -5
        self.total_reward = 0
        self.step_cnt = 0

    def _info(self):
        return {'occupy': self.occupy}

    def reset(self, *, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.total_reward = 0
        self.step_cnt = 0
        #重新随机选取位置
        self.position = generate_unique_coordinates(3)
        self.default_distance = compute_total_distance(self.position)
        self.last_distance = self.default_distance

        self.occupy = []
        for p in self.position:
            px = p[0]
            py = p[1]
            self.occupy.append(deepcopy(self.grid[px][py]))
        self.occupy = np.float32(self.occupy)
        self.obs = linear_scale(np.array(self.position,dtype=np.float32).flatten())
        #self.obs = np.concatenate([self.qubits, self.coupling, linear_scale(self.occupy)],dtype = np.float32)
        info = self._info()

        # trace = np.array(self.trace)
        # show_trace(trace.transpose())
        self.trace = []
        return self.obs , info

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        q = action[0]
        v = action[1]
        if v == 4:
            terminated = True
            reward = 0
        else:
            #执行 action
                #只计算坐标，并未真正 move
            x,y = self.move(v,self.position[q][0],self.position[q][1])
            if self.grid[x][y] not in self.occupy:
                #目标坐标无冲突
                self.position[q][0], self.position[q][1] = x,y
                self.occupy[q] = self.grid[x][y]
            reward = self.compute_reward(action)

        if self.total_reward <= self.stop_thresh \
                or reward <= self.stop_thresh \
                or self.step_cnt==self.max_step :
            terminated = True
        self.rs.append(reward)
        if self.debug:
            print('done = %r, reward = %r  info = %r \n' % (terminated, reward,self.occupy))
        self.trace.append(deepcopy(self.occupy))
        return self.obs, reward, terminated,truncated, self._info()


    def compute_reward(self,act):

        reward = self.stop_thresh
        #计算距离
        distance = compute_total_distance(self.position)
        k1 = (self.default_distance - distance) / self.default_distance
        k2 = (self.last_distance - distance) / self.last_distance

        if k2 > 0:
            reward = (math.pow((1 + k2), 2) - 1) * (1 + k1)
        elif k2 < 0:
            reward = -1 * (math.pow((1 - k2), 2) - 1) * (1 - k1)
        else:
            reward = 0

        self.last_distance = distance
        #计算是否满足连接性
        if meet_nn_constrain(self.nn):
            reward *= 10
        return reward

    def render(self):
        print('render')

    def close(self):
        self._close_env()

    def _close_env(self):
        pass
        #logger.info('_close_env')

    # 业务相关函数
    def move(self,direction,start_x,start_y):
        x,y = move_point(self.grid,direction,start_x,start_y)
        return (x,y)


