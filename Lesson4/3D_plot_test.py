# -*- coding: utf-8 -*-
# 2025-1-19 16:56:26
# Author: Andy

import numpy as np
import matplotlib.pyplot as plt

# 创建 3D 坐标系
fig = plt.figure()
# 使用 projection='3d' 创建 3D 坐标轴
ax = fig.add_subplot(111, projection='3d')

# 生成数据
x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)

# 绘制散点图
ax.scatter(x, y, z)

# 添加坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 显示图形
plt.show()
