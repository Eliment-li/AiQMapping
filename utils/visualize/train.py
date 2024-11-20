from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from config import ConfigSingleton

from utils.file.file_util import get_root_dir
args = ConfigSingleton().get_config()
#data[0]:  data[1]:
def show_train_metric(data,save = True,max_length = 20):

    # 创建一个图形和一个坐标轴
    fig, ax1 = plt.subplots()

    # Generate a colormap
    num_lines = len(data)
    colors = plt.cm.viridis(np.linspace(0, 1, num_lines))

    x1 = np.arange(len(data[0][:max_length]))
    y1 = data[0][:max_length]

    x2 = np.arange(len(data[1][:max_length]))
    y2 = data[1][:max_length]


    ax1.plot( x1, y1,  label = 'reward',color = '#5370c4',marker='o')
    ax1.set_xlabel('step')
    ax1.set_ylabel('reward',color='#5370c4')
    for x, y in zip(x1, y1):
        # Annotate each point with its value
        ax1.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        # Draw a vertical line from each point to the x-axis
        ax1.axvline(x=x, ymin=0, ymax=(y - plt.ylim()[0]) / (plt.ylim()[1] - plt.ylim()[0]), color=colors[0],
                    linestyle='--', linewidth=0.5)

    # for x, y in zip(x0, y0):
    #     # Annotate each point with its value
    #     plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
    #
    #     # Draw a vertical line from each point to the x-axis
    #     plt.axvline(x=x, ymin=0, ymax=(y - plt.ylim()[0]) / (plt.ylim()[1] - plt.ylim()[0]), color=colors[0],
    #                 linestyle='--', linewidth=0.5)



    # 创建第二个坐标轴，共享x轴
    ax2 = ax1.twinx()
    ax2.plot(x2, y2,label = 'distance', color ='#f16569',marker='v')
    ax2.set_ylabel('distance',color ='#f16569')
    for x, y in zip(x2, y2):
        # Annotate each point with its value
        ax2.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    # Add labels and legend
    plt.title('Train Metrics')

    if save:
        path = Path(get_root_dir())/ 'data' / 'result' /  (args.time_id + '.png')
        plt.savefig(path)
    # Show the plot
    plt.show()



if __name__ == '__main__':

    # Example usage:
    data = [
        [10, 20, 30, 40, 50],
        [0, 342, 200, 500, 600],
        [3, 4, 5, 6, 7]
    ]
    label = ['Metric 1', 'Metric 2', 'Metric 3']

    show_train_metric(data, label)
