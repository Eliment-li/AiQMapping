import math
import random
import re
import shutil
from copy import deepcopy
import numpy as np


valid_position = [[0, 1], [0, 3], [0, 5], [0, 7], [0, 9], [0, 11], [1, 0], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [2, 1], [2, 3], [2, 5], [2, 7], [2, 9], [2, 11], [3, 0], [3, 2], [3, 4], [3, 6], [3, 8], [3, 10], [4, 1], [4, 3], [4, 5], [4, 7], [4, 9], [4, 11], [5, 0], [5, 2], [5, 4], [5, 6], [5, 8], [5, 10], [6, 1], [6, 3], [6, 5], [6, 7], [6, 9], [6, 11], [7, 0], [7, 2], [7, 4], [7, 6], [7, 8], [7, 10], [8, 1], [8, 3], [8, 5], [8, 7], [8, 9], [8, 11], [9, 0], [9, 2], [9, 4], [9, 6], [9, 8], [9, 10], [10, 1], [10, 3], [10, 5], [10, 7], [10, 9], [10, 11]]

def generate_unique_coordinates(count):
    # 生成不重复的下标
    indices = random.sample(range(len(valid_position)), count)
    # 根据下标获取对应的值
    result = [deepcopy(valid_position[i]) for i in indices]
    return result

def unique_random_int(len,min=0,max=1):
    return random.sample(range(min,max), len)
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

# todo ,接口指定arr_min和 arr_max
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


def append2matrix(array, new_row):
    """
    把 1d array 加到 2d array 后面，如果太长则换行，太短则补0

    # demo:
    original_array = np.array([[1, 2, 3], [4, 5, 6]])
    new_row = [7, 8, 9, 10, 11, 12, 13]

    result_array = append2matrix(original_array, new_row)
    print(result_array)
    """
    length = len(array[0])
    # Convert the input array to a NumPy array if it isn't already
    array = np.array(array)

    # Determine the number of full rows we can get from new_row
    num_full_rows = len(new_row) // length
    remainder = len(new_row) % length

    # Create a list to store the rows to be added
    rows_to_add = []

    # Add full rows
    for i in range(num_full_rows):
        start_index = i * length
        end_index = start_index + length
        rows_to_add.append(new_row[start_index:end_index])

    # Add the remaining part of the row, padded with zeros if necessary
    if remainder > 0:
        last_row = new_row[num_full_rows * length:]
        last_row_padded = np.pad(last_row, (0, length - remainder), 'constant')
        rows_to_add.append(last_row_padded)

    # Convert the list of rows to a NumPy array
    rows_to_add = np.array(rows_to_add)

    # Append the new rows to the original array
    if array.size == 0:  # If the original array is empty
        return rows_to_add
    else:
        return np.vstack((array, rows_to_add))


def replace_last_n(matrix, replacement_array):
    '''
    用一维数组替换二维数组的最后 n 个元素。
    #demo
    matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
    replacement_array = [13, 14, 15, 16]

    # 调用函数
    new_matrix = replace_last_n(matrix, replacement_array)
    print(new_matrix)
    '''

    # 获取矩阵的行数和列数
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    n = len(replacement_array)

    # 确保 n 不超过矩阵的总元素数
    if n > rows * cols:
        raise ValueError("Replacement array is too long for the matrix.")

    # 从最后一个元素开始替换
    for i in range(n):
        # 计算二维数组中对应的行和列
        row_index = (rows * cols - n + i) // cols
        col_index = (rows * cols - n + i) % cols
        matrix[row_index][col_index] = replacement_array[i]

    return matrix



def parse_tensorboard(content):
    # 使用正则表达式搜索匹配的字符串
    pattern = r'tensorboard --logdir\s(.+)'
    result = re.findall(pattern, content)

    if result:
        matched_string = result[0][:-1]
        #tensorboard = matched_string[matched_string.find("tmp"):matched_string.find("driver_artifacts")]
        tensorboard = matched_string[matched_string.find("C"):matched_string.find("driver_artifacts")]
        return  tensorboard
    else:
        print("未找到匹配的 tensorboard")
        return  ''


# move a folder to another folder
def move_folder(src_folder, dest_folder):
    # Move the folder to the destination folder
    shutil.move(src_folder, dest_folder)

    # Print a message to indicate that the folder has been moved
    print(f"Folder '{src_folder}' has been moved to '{dest_folder}'.")
    return True
if __name__ == '__main__':
    ts= parse_tensorboard('To visualize your results with TensorBoard, run: `tensorboard --logdir C:/Users/Administrator/AppData/Local/Temp/ray/session_2024-11-19_11-11-06_708517_17204/artifacts/2024-11-19_11-11-15/PPO_2024-11-19_11-11-15/driver_artifacts`')
    print(ts)