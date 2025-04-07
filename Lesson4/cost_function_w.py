# -*- coding: utf-8 -*-
# 2025-1-20 18:43:25
# Author: Andy

import matplotlib.pyplot as plt
import dataset
import numpy as np

# 从数据中获取随机豆豆
m = 100
xs, ys = dataset.get_beans(m)

# 配置图像
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel('Bean Size')
plt.ylabel('Toxicity')
plt.xlim(0, 1)
plt.ylim(0, 1.5)

plt.scatter(xs, ys)

w = 0.1
b = 0.1

y_pre = w * xs + b

plt.plot(xs, y_pre)

plt.show()

fig = plt.figure()
# 使用 projection='3d' 创建 3D 坐标轴
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(0, 2)

ws = np.arange(-1, 2, 0.1)
bs = np.arange(-2, 2, 0.01)

# 创建新的绘图窗口
# plt.figure()

for b in bs:
    es = []
    for w in ws:
        y_pre = w * xs + b
        e = np.sum((ys - y_pre) ** 2) * (1 / m)
        es.append(e)
    # plt.plot(ws,es)
    ax.plot(ws, es, b, zdir="y")

# 添加坐标轴标签
ax.set_xlabel('b')
ax.set_ylabel('w')
ax.set_zlabel('Cost')

plt.show()

