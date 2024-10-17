import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch

def create_grid(rows, cols):
    grid = np.zeros((rows, cols))
    values = np.random.rand(rows, cols)  # Random values for each block
    num = 0
    for i in range(rows):
        for j in range(cols):
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                grid[i][j] = -1  # Invalid block
            else:
                grid[i][j] = num  # Valid block with a number
                num += 1
    return grid, values

def get_valid_positions(grid):
    positions = {}
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i][j] != -1:
                positions[f"Q{int(grid[i][j])}"] = (i, j)
    return positions

def plot_grid(grid, values, trajectories, text_color='black'):
    fig, ax = plt.subplots(dpi=250)
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, grid.shape[0] - 0.5)
    ax.set_aspect('equal')

    # Normalize the color map based on the values
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.viridis

    # Plot grid
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i][j] == -1:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='lightgrey'))
            else:
                color = cmap(norm(values[i, j]))
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=color))
                ax.text(j, i, f"Q{int(grid[i][j])}", ha='center', va='center', fontsize=8, color=text_color)

    # Plot trajectories with arrows
    color = ['#0D92F4','#F95454','#72BF78']
    color_idx = 0
    for trajectory in trajectories:
        for k in range(len(trajectory) - 1):
            start_pos = positions[f"Q{trajectory[k]}"]
            end_pos = positions[f"Q{trajectory[k+1]}"]
            arrow = FancyArrowPatch((start_pos[1], start_pos[0]), (end_pos[1], end_pos[0]),
                                    arrowstyle='-|>', color=color[color_idx], mutation_scale=10)
            ax.add_patch(arrow)
        color_idx += 1
        # Mark start and end points
        start_pos = positions[f"Q{trajectory[0]}"]
        end_pos = positions[f"Q{trajectory[-1]}"]
        ax.text(start_pos[1], start_pos[0], "Start", ha='right', va='bottom', fontsize=8, color='#3C3D37')
        ax.text(end_pos[1], end_pos[0], "End", ha='left', va='top', fontsize=8, color='#3C3D37')

    plt.gca().invert_yaxis()
    plt.show()

# Parameters
rows, cols = 10, 6
grid, values = create_grid(rows, cols)
positions = get_valid_positions(grid)

# Example trajectories
trajectories = [
    [0, 1, 3, 5],  # Example trajectory 1
    [2, 4, 6, 8]   # Example trajectory 2
]

# Plot with custom text color
plot_grid(grid, values, trajectories, text_color='white')
