import re
from collections import defaultdict
import matplotlib.pyplot as plt

# 示例输入数据，通常你会从文件或其他输入源读取
data = """
XY2M Q14 0.7853981633974483
X2M Q20
Y2M Q26
CZ Q14 Q20
B Q14 Q20 Q26
XY2M Q14 0.7853981633974483
XY2M Q20 0.7853981633974483
Y2P Q26
CZ Q20 Q26
B Q14 Q20 Q26
XY2P Q14 0.7853981633974483
Y2P Q20
XY2P Q26 0.7853981633974483
CZ Q14 Q20
B Q14 Q20 Q26
XY2P Q14 0.7853981633974483
Y2P Q20
XY2M Q26 0.7853981633974483
CZ Q20 Q26
B Q14 Q20 Q26
X2M Q14
X2M Q20
XY2P Q26 0.7853981633974483
CZ Q14 Q20
B Q14 Q20 Q26
XY2M Q14 0.7853981633974483
Y2M Q20
XY2P Q26 0.7853981633974483
CZ Q20 Q26
B Q14 Q20 Q26
XY2P Q14 0.7853981633974483
XY2P Q20 0.7853981633974483
XY2P Q26 0.7853981633974483
CZ Q14 Q20
B Q14 Q20 Q26
XY2P Q14 0.7853981633974483
XY2M Q20 0.7853981633974483
Y2P Q26
CZ Q20 Q26
B Q14 Q20 Q26
X2P Q14
Y2M Q20
XY2P Q26 0.7853981633974483
B Q14 Q20 Q26
M Q14
M Q20
M Q26
"""

# 使用正则表达式匹配CZ指令
pattern = re.compile(r'^CZ\s+(Q\d{2})\s+(Q\d{2})$')

# 字典用于统计每个Qxx Qxx组合的出现次数
counts = defaultdict(int)

# 遍历每一行
for line in data.strip().split('\n'):
    match = pattern.match(line.strip())
    if match:
        q1, q2 = match.groups()
        key = f"{q1} {q2}"
        counts[key] += 1

# 打印统计结果
for key, count in counts.items():
    print(f"{key} 出现 {count} 次")

# 准备数据用于绘制饼状图
labels = list(counts.keys())
sizes = list(counts.values())

# 创建饼状图
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('CZ Count')

# 显示图形
plt.axis('equal')  # 确保饼状图是圆形的
plt.show()
