import random

qmap = [
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

# Step 1: Collect all valid coordinates
valid_coordinates = []

for i in range(len(qmap)):
    for j in range(len(qmap[i])):
        if qmap[i][j] != -1:
            valid_coordinates.append((i, j))
print(valid_coordinates)
# Step 2: Randomly select three unique coordinates
selected_coordinates = random.sample(valid_coordinates, 3)

print(selected_coordinates)