import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.arange(0, 10, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建一个图形和一个坐标轴
fig, ax1 = plt.subplots()

# 在 ax1 上绘制第一个折线
ax1.plot(x, y1, 'g-', label='sin(x)')
ax1.set_xlabel('X 数据')
ax1.set_ylabel('sin(x)', color='g')

# 创建第二个坐标轴，共享 x 轴
ax2 = ax1.twinx()
ax2.plot(x, y2, 'b-', label='cos(x)')
ax2.set_ylabel('cos(x)', color='b')

# 添加图例
# 通过获取两个坐标轴的句柄和标签来创建一个综合图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

# 显示图形
plt.title('双折线图示例')
plt.show()
