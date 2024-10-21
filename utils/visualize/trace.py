import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.cm as cm
from sympy import pprint

from core.chip import QUBITS_ERROR_RATE


# Example usage
rows, cols = 11, 12
def get_values(rows, cols):
    vs = np.zeros((rows, cols))
    vi = 0
    for i in range(rows):
        for j in range(cols):
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                vs[i, j] = -1
            else:
                vs[i, j] = QUBITS_ERROR_RATE[vi]
                vi += 1
    return vs


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

colors = ['#0D92F4','#F95454','#72BF78','#FF7E07','#BE3BCE']

grid = create_grid(rows, cols)
values = get_values(11,12)
def show_trace( paths, grid =grid , values=values):
    rows, cols = grid.shape
    fig, ax = plt.subplots(figsize=(cols, rows), dpi=150)

    # Normalize values for color mapping
    cmap = cm.get_cmap('viridis', 256)

    for i in range(rows):
        for j in range(cols):
            value = grid[i, j]
            if value == -1:
                color = ('white')
            else:
                color = cmap(values[i][j])

            rect = patches.Rectangle((j, rows - i - 1), 1, 1, linewidth=0, edgecolor='none', facecolor=color)
            ax.add_patch(rect)

            if value != -1:
                ax.text(j + 0.5, rows - i - 0.5, f'Q{int(value)}', color='white', ha='center',fontsize = 18, va='center')

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




if __name__ == '__main__':
    trace=[[1,2,3,1,5],
 [44, 37, 37, 31],
 [49, 43,48, 54],
[41, 23,38, 44]]


    show_trace(paths=trace)

    # print(default_labels)
    #
    # for key in default_labels:
    #     print(key, default_labels[key])
    #     default_grid[key[0],key[1]] = default_labels[key]
    # default_grid = np.array(default_grid, dtype=int)
    #
    # for row in default_grid:
    #     print(", ".join(map(str, row)))