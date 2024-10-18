import math
import random

import numpy as np

valid_position = [[0, 1], [0, 3], [0, 5], [0, 7], [0, 9], [0, 11], [1, 0], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [2, 1], [2, 3], [2, 5], [2, 7], [2, 9], [2, 11], [3, 0], [3, 2], [3, 4], [3, 6], [3, 8], [3, 10], [4, 1], [4, 3], [4, 5], [4, 7], [4, 9], [4, 11], [5, 0], [5, 2], [5, 4], [5, 6], [5, 8], [5, 10], [6, 1], [6, 3], [6, 5], [6, 7], [6, 9], [6, 11], [7, 0], [7, 2], [7, 4], [7, 6], [7, 8], [7, 10], [8, 1], [8, 3], [8, 5], [8, 7], [8, 9], [8, 11], [9, 0], [9, 2], [9, 4], [9, 6], [9, 8], [9, 10], [10, 1], [10, 3], [10, 5], [10, 7], [10, 9], [10, 11]]

def generate_unique_coordinates(count):
    # 生成不重复的下标
    indices = random.sample(range(len(valid_position)), count)
    # 根据下标获取对应的值
    result = [valid_position[i] for i in indices]
    return result
def compute_total_distance(positions):
    # 初始化距离之和
    total_distance = 0

    # 计算所有点之间的距离之和
    num_points = len(positions)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            # 获取点的坐标
            x1, y1 = positions[i]
            x2, y2 = positions[j]

            # 计算两点之间的距离
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 将距离加到总和中
            total_distance += distance
    return total_distance

def data_normalization(data):
    """
    对输入数据进行Z-score标准化，然后应用Sigmoid函数进行归一化。
    参数:
    data (np.ndarray): 输入的一维NumPy数组。
    返回:
    np.ndarray: 归一化后的数据，范围在(0, 1)之间。
    """
    # 计算均值和标准差
    mean = np.mean(data)
    std = np.std(data)

    # Z-score标准化
    z_scores = (data - mean) / std

    # 定义Sigmoid函数
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # 应用Sigmoid函数
    normalized_data = sigmoid(z_scores)

    return normalized_data


def linear_scale(arr):
    """
    将数组中的值线性缩放到 [0, 1]。

    参数:
        arr (numpy.ndarray): 输入数组。

    返回:
        numpy.ndarray: 缩放后的数组。
    """
    arr_min = 0
    arr_max = 65

    # 防止除以零的情况
    if arr_max == arr_min:
        return np.zeros_like(arr)

    # 线性缩放公式
    scaled_arr = (arr - arr_min) / (arr_max - arr_min)
    return scaled_arr


# for i in range(100):
#     print(generate_unique_coordinates(3))