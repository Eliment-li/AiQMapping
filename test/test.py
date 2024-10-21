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


# 示例网格
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

# 计算相邻关系
neighbor_relations = get_neighbors(grid)

# 输出结果
for relation in neighbor_relations:
    print(relation)
