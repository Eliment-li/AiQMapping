import numpy as np
from sympy import pprint

from utils.common_utils import data_normalization
from utils.file.csv_util import read_df
from pathlib import Path
#path
directory  = Path('data')
qubits_error_path = directory / 'chip' / 'SingleQubit.csv'
coupling_path = directory / 'chip' / '2QCZXEB.csv'

def read_qubits_error_rate():
    data = read_df(relative_path=qubits_error_path)
    row_data = data.iloc[3, 1:67].to_numpy()
    row_data = [float(x) for x in row_data]
    return row_data

def read_coupling_score():
    data = read_df(relative_path=coupling_path)
    row_data = data.iloc[1, 1:90].to_numpy()
    row_data = [float(x) for x in row_data]
    return row_data


# Adjacency List of Chip
def read_adj_list():
    list = [[] for _ in range(66)]
    data = read_df(relative_path=coupling_path)
    column_names = data.columns.tolist()[1:-1]
    #print(column_names)

    for c in column_names:
        qa = int(c[1:3])
        qb = int(c[3:5])
        #print(f'{qa} to {qb}')
        #add qb to qa`s neighbor and vice versa
        # 这里优化为 set?
        if qb not in list[qa]:
            (list[qa]).append(qb)
        if qa not in list[qb]:
            (list[qb]).append(qa)


    return list

QUBITS_ERROR_RATE = data_normalization(read_qubits_error_rate())
COUPLING_SCORE = data_normalization(read_coupling_score())
ADJ_LIST = read_adj_list()


def meet_nn_constrain(nn:list[set()]):
    flag = True
    for i, s in enumerate(nn):
        if len(s) > 0:
            for v in s:
                # i,s 有依赖
                if not ADJ_LIST[i].__contains__(s):
                    flag = False
                    #print(f'{i} - {s} 不满足连接关系')
                    break
    return flag


def move_point(grid, direction,i,j):
    direction = int(direction)
    # 定义方向的坐标变换
    directions = {
        0: (-1, -1),#up left
        1: (-1, 1), #up right
        2: (1, -1), #down left
        3: (1, 1) #down right
    }

    # 获取方向的坐标变化
    if direction not in directions:
        raise ValueError("无效的方向")

    di, dj = directions[direction]

    # 计算新的坐标
    new_i, new_j = i + di, j + dj

    # 检查是否超出边界
    if new_i < 0 or new_i >= len(grid) or new_j < 0 or new_j >= len(grid[0]):
        # 超出边界，原地不动
        out_of_bounds = True
        new_i, new_j = i, j
    else:
        out_of_bounds = False
        # 检查目标坐标是否有效
        if grid[new_i][new_j] == -1:
            # 无效坐标，原地不动
            new_i, new_j = i, j

    # 返回新的坐标和标志位
    return (new_i, new_j)

grid = [
[-1, 0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5],
[6, -1, 7, -1, 8, -1, 9, -1, 10, -1, 11, -1],
[-1, 12, -1, 13, -1, 14, -1, 15, -1, 16, -1, 17],
[18, -1, 19, -1, 20, -1, 21, -1, 22, -1, 23, -1],
[-1, 24, -1, 25, -1, 26, -1, 27, -1, 28, -1, 29],
[30, -1, 31, -1, 32, -1, 33, -1, 34, -1, 35, -1],
[-1, 36, -1, 37, -1, 38, -1, 39, -1, 40, -1, 41],
[42, -1, 43, -1, 44, -1, 45, -1, 46, -1, 47, -1],
[-1, 48, -1, 49, -1, 50, -1, 51, -1, 52, -1, 53],
[54, -1, 55, -1, 56, -1, 57, -1, 58, -1, 59, -1],
[-1, 60, -1, 61, -1, 62, -1, 63, -1, 64, -1, 65],
]

