import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def create_grid(rows, cols):
    grid = np.zeros((rows, cols))
    num = 0

    for i in range(rows):
        for j in range(cols):
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                grid[i, j] = -1
            else:
                grid[i, j] = num
                num += 1
    return grid

colors = ['#0D92F4','#F95454','#72BF78']

def plot_grid(grid, values, paths):
    rows, cols = grid.shape
    fig, ax = plt.subplots(figsize=(cols, rows), dpi=150)

    # Normalize values for color mapping
    norm = plt.Normalize(min(values), max(values))
    cmap = plt.cm.viridis

    for i in range(rows):
        for j in range(cols):
            value = grid[i, j]
            if value == -1:
                color = ('#d1d3d7')
            else:
                color = cmap(norm(values[int(value)]))

            rect = patches.Rectangle((j, rows - i - 1), 1, 1, linewidth=0, edgecolor='none', facecolor=color)
            ax.add_patch(rect)

            if value != -1:
                ax.text(j + 0.5, rows - i - 0.5, f'Q{int(value)}', color='white', ha='center',fontsize = 16, va='center')

    # Plot paths
    idx = -1
    for path in paths:
        idx = idx+1
        color = colors[idx]
        path_coords = []
        for p in path:
            coords = np.argwhere(grid == p)
            if coords.size > 0:
                x, y = coords[0]
                path_coords.append((y + 0.5, rows - x - 0.5))

        if path_coords:
            path_coords = np.array(path_coords)
            #plt.plot(path_coords[:, 0], path_coords[:, 1], marker='', color=color, linewidth=4)

            # Add arrows
            for k in range(len(path_coords) - 1):
                arrow = patches.FancyArrowPatch(
                    path_coords[k], path_coords[k + 1],
                    arrowstyle='->', color=color, mutation_scale=40,connectionstyle = 'arc3,rad=-0.2',
                    lw = 4,
                )
                ax.add_patch(arrow)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.axis('off')
    plt.show()


# Example usage
rows, cols = 11, 12
grid = create_grid(rows, cols)

# Assigning random values to the valid blocks (Q0, Q1, ...)
values = np.random.rand(np.max(grid[grid != -1]).astype(int) + 1)

# Define paths using the block numbers
trace = [[1, 2, 3, 1, 5],
         [44, 37, 37, 31],
         [49, 43, 48, 54]]

plot_grid(grid, values, trace)
