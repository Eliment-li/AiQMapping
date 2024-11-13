import numpy as np
from sympy import pprint

from utils.common_utils import data_normalization, compute_total_distance, append2matrix
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


def cnt_meet_nn_constrain(nn,occupy):
    cnt = 0
    for i, s in enumerate(nn):
        # i= q left
        if len(s) <= 0:
            break
        for v in s:
            # 第i个比特,和比特v 有依赖
            if  ADJ_LIST[occupy[i]].__contains__(v):
                cnt += 1
                #print(f'{i} - {s} 满足连接关系')
    return cnt




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


def init_position_map():
    map = {}
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if not grid[i][j] == -1:
                map[int(grid[i][j])] = [i, j]
    return  map

def get_neighbors(grid):
    # 获取网格的行数和列数
    rows = len(grid)
    cols = len(grid[0])

    # 用于存储有效方块的编号
    valid_blocks = {}
    block_number = 0

    # 遍历整个网格，给有效方块编号
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != -1:
                valid_blocks[(i, j)] = block_number
                block_number += 1

    # 用于存储相邻关系
    neighbors = []

    # 定义四个对角方向的偏移量
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    # 遍历有效方块，查找其邻居
    for (i, j), block_id in valid_blocks.items():
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if (ni, nj) in valid_blocks:
                neighbor_id = valid_blocks[(ni, nj)]
                neighbors.append([block_id, neighbor_id])

    return neighbors

POSITION_MAP=init_position_map()
# 计算相邻关系
COUPLING_MAP = get_neighbors(grid)

#for attention
#表示芯片状态，用于 env 的state
# todo new_row 根据线路类型自动赋值
CHIPSTATE = append2matrix(grid,[0,0,0,0,0])



#计算距离
#备用，旋转坐标系
def rotate_grid_45_degrees(grid):
    original_rows = len(grid)
    original_cols = len(grid[0])
    new_size = original_rows + original_cols - 1
    new_grid = [[-2 for _ in range(new_size)] for _ in range(new_size)]

    coordinate_map = {}

    center = (new_size - 1) // 2

    for i in range(original_rows):
        for j in range(original_cols):
            if grid[i][j] != -1:
                new_row = i + j
                new_col = center + i - j
                new_grid[new_row][new_col] = grid[i][j]
                coordinate_map[grid[i][j]] = (new_row, new_col)

    return new_grid, coordinate_map

GRID_45 = [   [-2, -2, -2, -2, -2, -1, -2, -2, -2, -2, -2],
    [-2, -2, -2, -2, -2, 0, 6, -2, -2, -2, -2, -2],
    [-2, -2, -2, -2, -1, -1, -1, -2, -2, -2, -2],
    [-2, -2, -2, -2, 1, 7, 12, 18, -2, -2, -2, -2],
    [-2, -2, -2, -1, -1, -1, -1, -1, -2, -2, -2],
    [-2, -2, -2, 2, 8, 13, 19, 24, 30, -2, -2, -2],
    [-2, -2, -1, -1, -1, -1, -1, -1, -1, -2, -2],
    [-2, -2, 3, 9, 14, 20, 25, 31, 36, 42, -2, -2],
    [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
    [-2, 4, 10, 15, 21, 26, 32, 37, 43, 48, 54, -2],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 11, 16, 22, 27, 33, 38, 44, 49, 55, 60],
    [-2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2],
    [-2, 17, 23, 28, 34, 39, 45, 50, 56, 61, -2],
    [-2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2],
    [-2, -2, 29, 35, 40, 46, 51, 57, 62, -2, -2],
    [-2, -2, -2, -1, -1, -1, -1, -1, -1, -2, -2, -2],
    [-2, -2, -2, 41, 47, 52, 58, 63, -2, -2, -2],
    [-2, -2, -2, -2, -1, -1, -1, -1, -2, -2, -2, -2],
    [-2, -2, -2, -2, 53, 59, 64, -2, -2, -2, -2],
    [-2, -2, -2, -2, -2, -1, -1, -2, -2, -2, -2, -2],
    [-2, -2, -2, -2, -2, 65, -2, -2, -2, -2, -2]]

CORRD_45 = {}

#init CORRD_45
for i,row in enumerate(GRID_45):
    for j,val in enumerate(row):
        if val != -1:
            CORRD_45[val] = (i, j)


def chip_Qubit_distance(nn,occupy):
    dist = 0
    for i, s in enumerate(nn):
        # i= q left
        if len(s) == 0:
            continue
        for v in s:
            # 第i个比特,和比特v 有依赖
            # 计算 i, v  的距离
            # occupy[n]代表线路中 n号 qubit 在chip上对应的Qubit的编号
            x1,y1 = CORRD_45[occupy[i]]

            x2,y2 = CORRD_45[occupy[v]]
            dist += (abs(x1 - x2)/2) + abs(y1 - y2)
    return dist

