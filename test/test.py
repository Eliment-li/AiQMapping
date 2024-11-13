from pprint import pprint


def rotate_45(grid):
    num_rows = len(grid)
    num_cols = len(grid[0])
    max_diagonal_length = num_rows + num_cols - 1

    # Initialize the new grid with -2
    new_grid = [[] for _ in range(max_diagonal_length)]

    # Fill the new grid with the rotated elements
    for i in range(num_rows):
        for j in range(num_cols):
            new_grid[i + j].append(grid[i][j])

    # Determine the maximum row length for padding
    max_row_length = max(len(row) for row in new_grid)

    # Pad each row with -2 to make them all the same length
    for row in new_grid:
        while len(row) < max_row_length:
            row.insert(0, -2)
            row.append(-2)

    return new_grid


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

# Rotate and print the new grid
new_grid = rotate_45(grid)
print(repr(new_grid))
pprint(new_grid, width=80, indent=4, compact=True)
