import math


def calculate_total_distance(positions):
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


# 示例调用
positions = [
    [1, 0],
    [0, 2],
    [3, 0]
]

result = calculate_total_distance(positions)
print("所有点之间的距离之和为:", result)