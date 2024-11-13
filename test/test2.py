import math




def test_reward_function():
    #distance = [100,26.11,19.06,23.06,19.06]
    #distance = [24.36,26.11,19.06,23.06,19.06]
    distance = [2,1,2,1]
    default = distance[0]
    last = default
    total = 0
    for i,v in enumerate(distance):

        k1 = (default - v) / default
        k2 = (last - v) / last
        if k1==0:
            k1=0.5
        if k2 > 0:
            reward = (math.pow((1 + k2), 2) - 1) * math.fabs(k1)
        elif k2 < 0:
            reward = -1 * (math.pow((1 - k2), 2) - 1) * math.fabs( k1)
        else:
            reward = 0

        total = total*0.999 + reward
        print(f'{i}= {reward.__round__(4)}, total={total.__round__(2)}')

        last = v


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


# Original grid
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



if __name__ == '__main__':

    new_grid, coordinate_map = rotate_grid_45_degrees(grid)

    # Print the new grid
    for row in new_grid:
        print(row)

    # Print the coordinate map
    print("\nCoordinate Map:")
    for number, coordinates in sorted(coordinate_map.items()):
        print(f"Number {number}: {coordinates}")