from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.cm as cm
from sympy import pprint

from config import ConfigSingleton
from core.chip import QUBITS_ERROR_RATE
from utils.file.data import load_array
from utils.file.file_util import get_root_dir

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

colors = ['#F95454','#0D92F4','#72BF78','#FF7E07','#BE3BCE']

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
            elif value in [5,17,22,35,47]:
                color = ('lightgrey')
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

def show_result(result, grid =grid , values=values):
    rows, cols = grid.shape
    fig, ax = plt.subplots(figsize=(cols, rows), dpi=100)
    # Normalize values for color mapping
    cmap = cm.get_cmap('viridis', 256)
    for i in range(rows):
        for j in range(cols):
            value = grid[i, j]
            if value == -1:
                color = ('white')
            elif value in [5,17,22,35,47]:
                color = ('lightgrey')
            else:
                color = cmap(values[i][j])

            rect = patches.Rectangle((j, rows - i - 1), 1, 1, linewidth=0, edgecolor='none', facecolor=color)
            ax.add_patch(rect)

            if value != -1:
                ax.text(j + 0.5, rows - i - 0.5, f'Q{int(value)}', color='white', ha='center',fontsize = 18, va='center')

    # Plot paths
    idx = -1
    idx = idx + 1
    color = colors[idx]
    path_coords = []
    for p in result:
        coords = np.argwhere(grid == p)
        if coords.size > 0:
            x, y = coords[0]
            path_coords.append((y + 0.5, rows - x - 0.5))

    if path_coords:
        path_coords = np.array(path_coords)
        # plt.plot(path_coords[:, 0], path_coords[:, 1], marker='', color=color, linewidth=4)

        # Add arrows
        for k in range(len(path_coords)):
            if k <= len(path_coords) - 2:
                arrow = patches.FancyArrowPatch(
                    path_coords[k], path_coords[k + 1],
                    arrowstyle='->', color=color, mutation_scale=40, connectionstyle='arc3,rad=-0.2',
                    lw=4,
                )
                ax.add_patch(arrow)

                # Calculate the midpoint for text placement
            midpoint_x = path_coords[k][0]-0.35
            midpoint_y = path_coords[k][1]-0.35

            # Add text at the midpoint
            ax.text(
                midpoint_x, midpoint_y, f'{k}',
                fontsize=16, color='white', ha='center', va='center', rotation=0
            )

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.axis('off')
    path = Path(get_root_dir()) / 'data' / 'result' / (args.time_id + 'result.png')
    plt.savefig(path)
    if args.plot_result:
        plt.show()

args = ConfigSingleton().get_config()


if __name__ == '__main__':
    trace=[[2,56,26,39,32],
[33,56,26,39,32],
[33,56,26,39,32],
[27,56,26,39,32],
[27,56,26,39,32],
[27,56,26,39,32],
[27,56,26,39,32],
[27,56,26,33,32],
[27,56,26,33,32],
[27,56,26,33,32],
[27,56,26,21,32],
[33,56,26,21,32],
[33,56,26,21,39],
[33,27,26,21,39],
[33,27,32,21,39],
[33,27,32,21,39]]


    #loaded_array = load_array('array.txt')

    trace = np.array(trace).transpose()

    #show_trace(paths=trace)
    result =[9, 14, 2, 8, 4]

    show_result(result )
    # print(default_labels)
    #
    # for key in default_labels:
    #     print(key, default_labels[key])
    #     default_grid[key[0],key[1]] = default_labels[key]
    # default_grid = np.array(default_grid, dtype=int)
    #
    # for row in default_grid:
    #     print(", ".join(map(str, row)))