# -*- coding: utf-8 -*-
# 2025-1-20 18:31:16
# Author: Andy

import matplotlib.pyplot as plt
import dataset
import numpy as np

# 从数据中获取随机豆豆
m = 100
xs, ys = dataset.get_beans(m)

# 配置图像
plt.title("Size - Toxicity Function", fontsize=12)
plt.xlabel('Bean Size')
plt.ylabel('Toxicity')

# 豆豆毒性散点图
plt.scatter(xs, ys)

# 预测函数
w = 0.1
b = 0.1
y_pre = w * xs + b

# 预测函数图像
plt.plot(xs, y_pre)

# 显示图像
plt.show()

# 代价函数
ws = np.arange(-1, 2, 0.1)
bs = np.arange(-2, 2, 0.1)

# 把 ws 和 bs 变成一个网格矩阵
ws, bs = np.meshgrid(ws, bs)

es = 0
# 因为 ws 和 bs 已经变成了网格矩阵了
# 一次性带入全部计算，我们需要一个一个的算
for i in range(m):
    y_pre = ws * xs[i] + bs
    e = (ys[i] - y_pre) ** 2
    es += e
es = es / m

fig = plt.figure()
# 使用 projection='3d' 创建 3D 坐标轴
ax = fig.add_subplot(111, projection='3d')

ax.set_zlim(0, 2)

# plot_surface 函数绘制曲面
# cmap='rainbow' 表示彩虹图（用不同的颜色表示不同值）
ax.plot_surface(ws, bs, es, cmap='rainbow')

# 添加坐标轴标签
ax.set_xlabel('b')
ax.set_ylabel('w')
ax.set_zlabel('Cost')

# 显示图像
plt.show()

