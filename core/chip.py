import numpy as np

from utils.file.csv_util import read_df
from pathlib import Path
#path
directory  = Path('data')
qubits_error_path = directory / 'chip' / 'SingleQubit.csv'
coupling_path = directory / 'chip' / '2QCZXEB.csv'

def read_qubits_error_rate():
    data = read_df(relative_path=qubits_error_path)
    row_data = data.iloc[2, 1:64].to_numpy()
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

QUBITS_ERROR_RATE = read_qubits_error_rate()
COUPLING_SCORE = read_coupling_score()
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




def move_point(grid, direction,start_x,start_y):
    """
    移动坐标点并返回新的坐标。

    :param grid: int, n*n 数组的大小
    :param start: tuple, 初始点的坐标 (row, column)
    :param direction: str, 移动的方向，'up', 'down', 'left', 'right' 之一
    :return: tuple, 移动后的坐标 (row, column)
    """
    row_size = len(grid)
    column_size = len(grid[0])
    # 解构初始坐标
    row, col = start_x, start_y

    # 根据方向调整坐标
    if direction == 'up':
        row -= 1
    elif direction == 'down':
        row += 1
    elif direction == 'left':
        col -= 1
    elif direction == 'right':
        col += 1
    else:
        raise ValueError("Invalid direction. Use 'up', 'down', 'left', or 'right'.")

    # 检查边界条件
    if row < 0:
        row = 0
    elif row >= row_size:
        row = row_size - 1

    if col < 0:
        col = 0
    elif col >= column_size:
        col = column_size - 1

    return (row, col)

qmap = [
    [0, 1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10, 11],
    [12, 13, 14, 15, 16, 17],
    [18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29],
    [30, 31, 32, 33, 34, 35],
    [36, 37, 38, 39, 40, 41],
    [42, 43, 44, 45, 46, 47],
    [48, 49, 50, 51, 52, 53],
    [54, 55, 56, 57, 58, 59],
    [60, 61, 62, 63, 64, 65],
]
