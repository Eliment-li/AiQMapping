from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.file.file_util import get_root_dir


def show_train_metric(data, label,save = True,max_length = 20):
    #对 dist 进行归一化
    data[1] = data[1]/(np.max(data[1])/np.max(data[0]))
    data[2] = data[2]/(np.max(data[2])/np.max(data[0]))

    # Check if the data and label lengths match
    if len(data) != len(label):
        raise ValueError("The number of labels must match the number of data rows.")

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Generate a colormap
    num_lines = len(data)
    colors = plt.cm.viridis(np.linspace(0, 1, num_lines))

    # Plot each row of the data
    # for i, (row, lbl) in enumerate(zip(data, label)):
    #     plt.plot(row[0:max_length], label=lbl, color=colors[i])
    for i, (row, lbl) in enumerate(zip(data, label)):
        x_values = np.arange(len(row[:max_length]))
        y_values = row[:max_length]
        plt.plot(x_values, y_values, label=lbl, color=colors[i],marker='o')

        for x, y in zip(x_values, y_values):
            # Annotate each point with its value
            plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

            # Draw a vertical line from each point to the x-axis
            plt.axvline(x=x, ymin=0, ymax=(y - plt.ylim()[0]) / (plt.ylim()[1] - plt.ylim()[0]), color=colors[i],
                        linestyle='--', linewidth=0.5)

    # Add labels and legend
    plt.xlabel('step')
    plt.ylabel('Value')
    plt.title('Train Metrics')
    plt.legend()
    if save:
        p = Path(get_root_dir())
        datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        path = p / 'data' / 'result' /  (datetime_str + '.png')
        plt.savefig(path)
    # Show the plot
    plt.show()




# Example usage:
data = [
    [10, 20, 30, 40, 50],
    [200, 300, 400, 500, 600],
    [3, 4, 5, 6, 7]
]
label = ['Metric 1', 'Metric 2', 'Metric 3']

show_train_metric(data, label)
