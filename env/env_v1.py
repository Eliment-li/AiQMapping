import math
import datetime

import gymnasium as gym
import numpy
import numpy as np

from gymnasium import spaces, register
from gymnasium.spaces import MultiBinary, MultiDiscrete,Discrete
from gymnasium.spaces.utils import flatten_space
from qiskit_aer import AerSimulator
from loguru import logger
import warnings

# from utils.concurrent_set import  SingletonMap
# from utils.file.csv_util import CSVUtil
# from utils.file.file_util import FileUtil
# from utils.graph_util import GraphUtil as gu, GraphUtil
# from config import get_args, ConfigSingleton
import os
import traceback

os.environ["SHARED_MEMORY_USE_LOCK"] = '1'

from shared_memory_dict import SharedMemoryDict
simulator = AerSimulator()
'''
v1
'''
warnings.filterwarnings("ignore")
class CircuitEnv_v1(gym.Env):
    def __init__(self, render_mode=None,kwargs = {'debug':False},env_config=None):
        self.debug = kwargs.get('debug')

        self.mem_cnt = 0
        self.all_cnt=0
        self.hit_rate=[]
        # circuit 相关变量
        qasm = SharedMemoryDict(name='tokens',size=1024).get('qasm')
        self.circuit = self.get_criruit(qasm)

        self.qubit_nums = len(self.circuit.qubits)

        obs_size = int((self.qubit_nums * self.qubit_nums - self.qubit_nums ) / 2)
        self.observation_space = flatten_space(spaces.Box(0,1,(1,obs_size),dtype=np.uint8,))
        self.action_space = MultiDiscrete([self.qubit_nums , self.qubit_nums])

        self.max_step = 100
        self.max_edges=4
        self.stop_thresh = -2

    def _get_info(self):
        return {'info':'this is info'}

    def reset(self, *, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        info = self._get_info()
        return self._get_obs(), info

    def step(self, action):
        self.step_cnt += 1
        self.all_cnt += 1

        reward,observation = self._get_rewards(action)
        info = self._get_info()

        terminated = False
        truncated = False
        if self.total_reward <= self.stop_thresh \
                or reward == self.stop_thresh \
                or self.step_cnt==self.max_step :
            terminated = True

        return observation, reward, terminated,truncated, info

    def render(self):
        print('render')

    def close(self):
        self._close_env()

    def _get_obs(self):
        return ''

    def _get_info(self):
        return {"info":"this is info"}


    def _get_rewards(self,act):
        reward = 0
        if self.debug:
            print('action = %r,  step=%r , score=%r ,reward=%r  \n obs=%r,'%(act,self.step_cnt,score,reward,self.obs))


        return reward,self._get_obs()

    def _close_env(self):
        logger.info('_close_env')
        #self.log_hit_rate()


if __name__ == '__main__':
    pass


