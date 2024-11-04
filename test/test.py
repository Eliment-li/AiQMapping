# 定义二维数组 (m x n)
matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
]

# 定义一维数组
replacement_array = [13, 14, 15, 16,17]

# 确定替换的起始位置
# 这里假设我们要替换的是最后一行的元素
rows = len(matrix)
cols = len(matrix[0])
n = len(replacement_array)

# 确保 n <= rows * cols
if n <= rows * cols:
    # 从最后一个元素开始替换
    for i in range(n):
        # 计算二维数组中对应的行和列
        row_index = (rows * cols - n + i) // cols
        col_index = (rows * cols - n + i) % cols
        matrix[row_index][col_index] = replacement_array[i]

print(matrix)