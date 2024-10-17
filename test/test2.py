import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import random

def create_grid(rows, cols):
    grid = np.zeros((rows, cols))
    values = np.random.rand(rows, cols)  # Random values for demonstration
    for i in range(rows):
        for j in range(cols):
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                grid[i, j] = -1  # Invalid block
            else:
                grid[i, j] = values[i, j]  # Assign a random value for valid blocks
    return grid

def label_blocks(grid):
    label = 0
    labels = {}
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] != -1:
                labels[f'Q{label}'] = (i, j)
                label += 1
    return labels

def visualize_grid(grid, labels, paths):
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, grid.shape[0] - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')  # Remove border

    # Define color map
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=1)

    # Draw the grid
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == -1:
                color = 'gray'
            else:
                color = cmap(norm(grid[i, j]))  # Get color from colormap based on value
            rect = patches.Rectangle((j-0.5, grid.shape[0]-i-1-0.5), 1, 1, linewidth=0, edgecolor='none', facecolor=color)
            ax.add_patch(rect)

    # Draw labels
    for label, (i, j) in labels.items():
        ax.text(j, grid.shape[0]-i-1, label, ha='center', va='center', color='black')  # Customize text color here

    # Draw paths
    for path in paths:
        path_coords = [labels[f'Q{num}'] for num in path]
        path_coords = [(j, grid.shape[0]-i-1) for i, j in path_coords]
        xs, ys = zip(*path_coords)
        color = [random.random() for _ in range(3)]
        ax.plot(xs, ys, marker='o', color=color, linewidth=2)
        for k in range(len(xs) - 1):
            ax.annotate('', xy=(xs[k+1], ys[k+1]), xytext=(xs[k], ys[k]),
                        arrowprops=dict(facecolor=color, shrink=0.05, width=2, headwidth=8))

    plt.show()

# Parameters
rows = 10
cols = 6
grid = create_grid(rows, cols)
labels = label_blocks(grid)

# Example paths
paths = [
    [0, 1, 2, 3],   # Path from Q0 to Q3
    [5, 6, 7],      # Path from Q5 to Q7
]

visualize_grid(grid, labels, paths)
