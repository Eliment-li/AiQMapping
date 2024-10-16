import math
import datetime
from copy import copy

import gymnasium as gym
import numpy as np

from gymnasium import spaces, register
from gymnasium.spaces import MultiBinary, MultiDiscrete,Discrete
from gymnasium.spaces.utils import flatten_space
from qiskit_aer import AerSimulator
from loguru import logger
import warnings
from typing import Optional
# from utils.concurrent_set import  SingletonMap
# from utils.file.csv_util import CSVUtil
# from utils.file.file_util import FileUtil
# from utils.graph_util import GraphUtil as gu, GraphUtil
# from config import get_args, ConfigSingleton
import os
import traceback

from core.chip import QUBITS_ERROR_RATE, move_point, qmap, COUPLING_SCORE, ADJ_LIST, meet_nn_constrain
from utils.circuits_util import qubits_nn_constrain
from utils.common_utils import calculate_total_distance

os.environ["SHARED_MEMORY_USE_LOCK"] = '1'

from shared_memory_dict import SharedMemoryDict
simulator = AerSimulator()
'''
v1
'''
warnings.filterwarnings("ignore")
class CircuitEnv_v1(gym.Env):
    def __init__(self, config: Optional[dict] = None):
        self.debug = config.get('debug')

        # circuit 变量
        self.circuit = 'XEB_3_qubits_8_cycles_circuit.txt'
        self.qubit_nums = 3

        #chip 变量
        #记录 qubits 当前的 position
        self.position = [
            [0,0]
            [0,0]
            [0,0]
        ]
        self.nn = qubits_nn_constrain()
        self.grid = copy(qmap)

        self.qubits = QUBITS_ERROR_RATE
        self.coupling= COUPLING_SCORE
        # 被占据的qubit
        self.occupy = []
        obs_size = len(self.qubits) + len(self.coupling) + self.qubit_nums

        # todo 先试试 flatten, 后面尝试直接用 spaces.Box
        self.observation_space = flatten_space(spaces.Box(0,1,(1,obs_size),dtype=np.float16,))

        self.obs = []

        #todo: action_space 开的大一点，只取前 n 位有用的，以适应不同线路
        # 0,1,2,3 = 上 下 左 右
        # 先尝试寻路法，每次只走一步
        self.action_space = MultiDiscrete([self.qubit_nums , 4])

        #stop conditions
        self.max_step = 10000
        self.stop_thresh = -2

    def _get_info(self):
        return ''

    def reset(self, *, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.obs = self.qubits + self.coupling + self.occupy
        info = self._info()
        return self._get_obs(), info

    def step(self, action):
        #执行 action
        for i, v in enumerate(action):
            x,y = self.move(v,self.position[i][0],self.position[i][1])
            #更新坐标
            self.position[i][0], self.position[i][1] = x,y
            #更新 occupy 数组
            self.occupy[i] = self.grid[x][y]

        reward = self.compute_reward(action)

        terminated = False
        truncated = False
        if self.total_reward <= self.stop_thresh \
                or reward == self.stop_thresh \
                or self.step_cnt==self.max_step :
            terminated = True

        return self.obs, reward, terminated,truncated, self._info()


    def compute_reward(self,act):
        reward = self.stop_thresh

        #计算距离
        distance = calculate_total_distance(self.position)
        #计算是否满足连接性
        meet_nn_constrain()

        k1 = (self.default_distance - distance) / self.default_distance
        k2 = (self.last_score - distance) / self.last_distance

        if k2 > 0:
            reward = (math.pow((1 + k2), 2) - 1) * (1 + k1)
        elif k2 < 0:
            reward = -1 * (math.pow((1 - k2), 2) - 1) * (1 - k1)
        else:
            reward = 0
        self.last_distance = distance


        return reward

    def render(self):
        print('render')

    def close(self):
        self._close_env()

    def _get_obs(self):
        return ''

    def _close_env(self):
        logger.info('_close_env')

    # 业务代码
    def move(self,direction,start_x,start_y):
        map = {
            0:'up',
            1:'down',
            2:'left',
            3:'right',
        }
        x,y = move_point(self.grid,map[direction],start_x,start_y)
        return (x,y)







if __name__ == '__main__':
    pass


