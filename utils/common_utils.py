import math
import random

valid_position = [[0, 1], [0, 3], [0, 5], [0, 7], [0, 9], [0, 11], [1, 0], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [2, 1], [2, 3], [2, 5], [2, 7], [2, 9], [2, 11], [3, 0], [3, 2], [3, 4], [3, 6], [3, 8], [3, 10], [4, 1], [4, 3], [4, 5], [4, 7], [4, 9], [4, 11], [5, 0], [5, 2], [5, 4], [5, 6], [5, 8], [5, 10], [6, 1], [6, 3], [6, 5], [6, 7], [6, 9], [6, 11], [7, 0], [7, 2], [7, 4], [7, 6], [7, 8], [7, 10], [8, 1], [8, 3], [8, 5], [8, 7], [8, 9], [8, 11], [9, 0], [9, 2], [9, 4], [9, 6], [9, 8], [9, 10], [10, 1], [10, 3], [10, 5], [10, 7], [10, 9], [10, 11]]

def generate_unique_coordinates(count):
    return  random.sample(valid_position, count)
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

# arr = generate_unique_coordinates(3,10,10)
# arr[1][0] = 0
# arr[1][1] = 0
# arr[2][0]  = 1
# arr[2][1] = 1
#
# print(arr)
