import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.cm as cm



def create_grid(rows, cols):
    grid = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                grid[i, j] = -1
            else:
                grid[i, j] = np.random.rand()
    return grid

def assign_labels(grid):
    labels = {}
    counter = 0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] != -1:
                labels[(i, j)] = counter
                counter += 1
    return labels

# Example usage
rows, cols = 11, 12
default_grid = create_grid(rows, cols)
default_labels = assign_labels(default_grid)

# Define trajectories as lists of label numbers
# trajectories = [
#     [0, 1, 3, 5, 7],
#     [2, 4, 6, 8, 10]
# ]

def visualize_grid(trajectories,grid=default_grid, labels=default_labels):
    # Increase the size and resolution of the figure
    fig, ax = plt.subplots( dpi=250)
    cmap = cm.get_cmap('viridis', 256)

    # Plot each cell
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            color = 'white' if grid[i, j] == -1 else cmap(grid[i, j])
            rect = patches.Rectangle((j, grid.shape[0] - i - 1), 1, 1, linewidth=0, edgecolor='none', facecolor=color)
            ax.add_patch(rect)
            if grid[i, j] != -1:
                ax.text(j + 0.5, grid.shape[0] - i - 0.5, f'Q{labels[(i, j)]}', ha='center', va='center', fontsize=8, color='white')

    # Plot the trajectories
    colors = ['#0D92F4','#F95454','#72BF78']#cm.rainbow(np.linspace(0, 1, len(trajectories)))
    for idx, trajectory in enumerate(trajectories):
        coords = [key for key, value in labels.items() if value in trajectory]
        for k in range(len(coords) - 1):
            start = coords[k]
            end = coords[k + 1]
            # Calculate directional offsets
            dx = end[1] - start[1]
            dy = start[0] - end[0]
            norm = np.sqrt(dx**2 + dy**2)
            # Shorten the arrow by 0.2 units at both ends
            offset = 0.2 / norm
            start_x = start[1] + 0.5 + dx * offset
            start_y = grid.shape[0] - start[0] - 0.5 + dy * offset
            end_x = end[1] + 0.5 - dx * offset
            end_y = grid.shape[0] - end[0] - 0.5 - dy * offset
            ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y,
                     head_width=0.2, head_length=0.2, fc=colors[idx], ec=colors[idx])

    # Set limits and remove axes
    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(0, grid.shape[0])
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()



if __name__ == '__main__':
    trajectories=[
        [1,2,3],
        [4,5,6]
    ]
    visualize_grid(trajectories=trajectories)
