import math
import datetime
from copy import copy,deepcopy
from logging import lastResort
from pprint import pprint

import gymnasium as gym
import numpy as np
from gymnasium import  register
from gymnasium.spaces import MultiBinary, MultiDiscrete, Discrete, Box
from gymnasium.spaces.utils import flatten_space
from qiskit_aer import AerSimulator
from loguru import logger
import warnings
from typing import Optional
import os

from config import ConfigSingleton
from core.chip import QUBITS_ERROR_RATE, move_point, grid, COUPLING_SCORE, POSITION_MAP, \
    cnt_meet_nn_constrain
import utils.circuits_util as cu
from utils.common_utils import compute_total_distance, generate_unique_coordinates, data_normalization, linear_scale
from utils.visualize.trace import show_trace
from env.reward_function import RewardFunction
os.environ["SHARED_MEMORY_USE_LOCK"] = '1'
args = ConfigSingleton().get_config()
rfunctions = RewardFunction()
simulator = AerSimulator()
'''
v9 修改 meet nn constrain 算法
'''
warnings.filterwarnings("ignore")
class CircuitEnv_v9(gym.Env):
    def __init__(self, config: Optional[dict] = None):
        self.debug = False #config.get('debug')
        self.trace = []
        self.rs = []
        # circuit 变量
        self.qubit_nums = 5
        self.circuit = 'XEB_'+str(self.qubit_nums)+'_qubits_8_cycles_circuit.txt'

        #chip 变量
        self.position =generate_unique_coordinates(self.qubit_nums)
        self.nn = cu.qubits_nn_constrain(self.circuit)
        self.grid = copy(grid)
        self.max_nn_meet = 0
        # 被占据的qubit，用 Q序号为标识
        self.occupy = []
        for p in self.position:
            px = p[0]
            py = p[1]
            self.occupy.append(deepcopy(self.grid[px][py]))

        self.qubits = np.float32(QUBITS_ERROR_RATE)
        self.coupling= np.float32(COUPLING_SCORE)

        obs_size = self.qubit_nums+1*2
        #先试试 flatten, 后面尝试直接用 spaces.Box
        high = np.array([66] * self.qubit_nums)
        self.observation_space = Box(0, 1, (66+self.qubit_nums,), np.float32)

        self.obs = np.array(self.occupy).astype(int)
        self.action_space = MultiDiscrete([(self.qubit_nums+1), 65])

        self.default_distance = compute_total_distance(self.position)
        self.last_distance = self.default_distance

        #stop conditions
        self.max_step = -1
        self.stop_thresh = -100
        self.total_reward = 0
        self.step_cnt = 0

    def _info(self):
        return {'occupy': self.occupy}

    def reset(self, *, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.max_nn_meet = 0
        self.total_reward = 0
        self.step_cnt = 0
        #重新随机选取位置
        # todo  直接从 position map 中选就行
        self.position = generate_unique_coordinates(self.qubit_nums)
        self.default_distance = compute_total_distance(self.position)
        self.last_distance = self.default_distance

        #初始化错误信息
        error = 0
        for v in self.occupy:
            error += QUBITS_ERROR_RATE[v]
        self.default_error = error
        self.last_error = error

        self.occupy = []
        for p in self.position:
            px = p[0]
            py = p[1]
            self.occupy.append(deepcopy(self.grid[px][py]))

        info = self._info()

        # trace = np.array(self.trace)
        # show_trace(trace.transpose())
        self.trace = []
        return self.get_obs() , info

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        q1 = action[0]
        q2 = action[1]
        #终止条件
        if q1 == 5:
            terminated = True
            reward = 0
        else:
            #执行 远距离移动 q1->q2
            x,y = POSITION_MAP[int(q2)][0],POSITION_MAP[int(q2)][1]
            if not np.any(np.all(self.position == np.array([x,y]), axis=1)):
                #目标坐标无冲突
                self.position[q1][0], self.position[q1][1] = x,y
                #self.occupy[q1] = self.grid[x][y]
                self.occupy[q1] = q2

                reward = self.compute_reward(action)
        #stop conditions
            # 计算是否满足连接性
        cnt =  cnt_meet_nn_constrain(self.nn,self.occupy)
        if cnt > self.max_nn_meet:
            reward =  4 * cnt
            self.max_nn_meet = cnt
        if cnt == len(self.nn):
            reward  =  2*cnt
            Terminated =True
        if reward == 0:
            reward = -0.01

        if self.total_reward <= self.stop_thresh \
                or reward <= self.stop_thresh \
                or self.step_cnt==self.max_step :
            terminated = True

        self.rs.append(reward)

        if self.debug:
            print('done = %r, reward = %r  info = %r \n' % (terminated, reward,self.occupy))

        self.trace.append(deepcopy(self.occupy))
        return self.get_obs(), reward, terminated,truncated, self._info()

    def get_obs(self):
        self.obs = np.concatenate((QUBITS_ERROR_RATE, data_normalization([1, 2, 3, 4, 5]))).astype(np.float32)
        return deepcopy(self.obs)

    def compute_reward(self,act):
        reward = 0
        rf_name = f"rfv{args.reward_function_version}"
        function_to_call = getattr(rfunctions, rf_name, None)
        if callable(function_to_call):
            reward = function_to_call(self,act)
        else:
            print(f"Function {rf_name} does not exist.")
        return  reward

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


if __name__ == '__main__':
    print(Box(0, 1, (66,), np.float32).sample())