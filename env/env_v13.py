import math
import datetime
from copy import copy,deepcopy
from logging import lastResort
from pathlib import Path
from pprint import pprint

import gymnasium as gym
import numpy as np
from gymnasium import  register
from gymnasium.spaces import MultiBinary, MultiDiscrete, Discrete, Box
from gymnasium.spaces.utils import flatten_space
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from loguru import logger
import warnings
from typing import Optional
import os

from config import ConfigSingleton
from core.chip import QUBITS_ERROR_RATE, move_point, grid, COUPLING_SCORE, POSITION_MAP, \
    cnt_meet_nn_constrain, chip_Qubit_distance
import utils.circuits_util as cu
from utils.code_conversion import QCIS_2_QASM
from utils.common_utils import data_normalization, unique_random_int, append2matrix, replace_last_n
from utils.file.file_util import read_all
from utils.visualize.trace import show_trace
from env.reward_function import RewardFunction
os.environ["SHARED_MEMORY_USE_LOCK"] = '1'
args = ConfigSingleton().get_config()
rfunctions = RewardFunction()
simulator = AerSimulator()
'''
v13 继承v12 更改部分策略
'''
warnings.filterwarnings("ignore")
class CircuitEnv_v13(gym.Env):
    def __init__(self, config: Optional[dict] = None):
        self.debug = False #config.get('debug')
        self.trace = []
        #save reward
        self.rs = []
        # circuit 变量
        self.qubit_nums = 5
        self.circuit = 'XEB_'+str(self.qubit_nums)+'_qubits_8_cycles_circuit.txt'
        path = Path(args.circuit_path) / self.circuit
        RE_LABEL_CIRCUIT = cu.reassign_qxx_labels(read_all(path))
        QASM_STR = QCIS_2_QASM(RE_LABEL_CIRCUIT)
        self.QiskitCircuit = QuantumCircuit.from_qasm_str(qasm_str=QASM_STR)

        #chip 变量
        #todo
        self.nn = cu.qubits_nn_constrain(self.circuit)
        self.grid = deepcopy(grid)
        # 被占据的qubit，用 Q序号为标识
        self.occupy = unique_random_int(self.qubit_nums, 0, 65)

        # init an array from 0 to self.qubit_nums - 1

        self.qubits = np.float32(QUBITS_ERROR_RATE)
        self.coupling= np.float32(COUPLING_SCORE)

        # -1 在 uint8 下 变为255 标识失效位置
        self.observation_space = Box(0, 255, (12,12), dtype=np.uint8)
        self.obs = append2matrix(self.grid, self.occupy)
        self.action_space = MultiDiscrete([(self.qubit_nums), 65])

        self.default_distance = chip_Qubit_distance(nn = self.nn,occupy=self.occupy)
        self.last_distance = self.default_distance

        #stop conditions
        self.max_step = 20
        self.stop_thresh = -100
        self.total_reward = 0
        self.step_cnt = 0


    def _info(self):
        return {'occupy': self.occupy,
                'distance': self.last_distance,
                'nn': 0,
                }

    def reset(self, *, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.total_reward = 0
        self.step_cnt = 0

        #重新随机选取位置
        self.occupy = unique_random_int(self.qubit_nums, 0, 65)
        self.obs = replace_last_n(self.obs ,self.occupy)
        self.default_distance = chip_Qubit_distance(nn=self.nn, occupy=self.occupy)
        self.last_distance = self.default_distance

        #初始化错误信息
        error = 0
        for v in self.occupy:
            error += QUBITS_ERROR_RATE[v]
        self.default_error = error
        self.last_error = error


        info = self._info()

        # trace = np.array(self.trace)
        # show_trace(trace.transpose())
        self.trace = []
        return self.get_obs() , info

    def step(self, action):
        self.step_cnt += 1
        reward = 0
        truncated = False

        q = action[0] #logical qubit
        Q = action[1] #physics qubit
        #终止条件
        if q == self.qubit_nums or self.step_cnt >= self.max_step:
            terminated = True
            truncated = True
        else:
            #move logic qubit to physics Qubit

            if not Q in self.occupy:
                # q2位置为空,直接占据
                self.occupy[q] = Q
            else:
                # q2位置不为空,交换位置
                qold = self.occupy.index(Q)
                temp = self.occupy[q]
                self.occupy[q] = Q
                self.occupy[qold] = temp
            #更新 obs
            self.obs = replace_last_n(self.obs ,self.occupy)
            reward,terminated = self.compute_reward(action)
        #stop conditions


        if self.total_reward <= self.stop_thresh \
                or reward <= self.stop_thresh \
                 :
            terminated = True
            truncated = True

        self.rs.append(reward)

        if self.debug:
            print('done = %r, reward = %r  info = %r \n' % (terminated, reward,self.occupy))

        self.trace.append(deepcopy(self.occupy))
        return self.get_obs(), reward, terminated,truncated, self._info()

    def get_obs(self):
        return self.obs

    def compute_reward(self,act):
        reward = 0
        rf_name = f"rfv{args.reward_function_version}"
        function_to_call = getattr(rfunctions, rf_name, None)
        if callable(function_to_call):
            reward,terminated = function_to_call(self,act)
        else:
            print(f"Function {rf_name} does not exist.")
        return  reward,terminated

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