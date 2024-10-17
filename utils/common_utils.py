import math
import random

def generate_unique_coordinates(count,max_X,max_Y):
    coordinates = set()

    while len(coordinates) < count:
        x = random.randint(0, max_X - 1)
        y = random.randint(0, max_Y - 1)
        coordinates.add((x, y))

    return [list(coord) for coord in coordinates]

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
