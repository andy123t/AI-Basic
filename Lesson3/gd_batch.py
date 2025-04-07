# -*- coding: utf-8 -*-
# 2024-04-21 18:40:15
# Author: Andy

import dataset
from matplotlib import pyplot as plt
import numpy as np

n = 100

xs,ys = dataset.get_beans(n)

# 配置图像
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean Size") #设置横坐标的名字
plt.ylabel("Toxicity") #设置纵坐标的名字

plt.scatter(xs,ys)

w = 0.1

y_pre = w*xs

plt.plot(xs,y_pre)

# plt.show()

alpha = 0.1
for _ in range(100):
    # 代价函数:e=(y-w*x)^2=x^2*w^2+(-2x*y)*w+y^2
    # a=x^2
    # b=-2x*y
    # 求解斜率: k=2aw+b
    k = 2*np.sum(xs**2)*w + np.sum(-2*xs*ys)
    k = k/100
    w = w - alpha*k
    y_pre = w*xs
    plt.clf() #清空窗口
    plt.xlim(0,1)
    plt.ylim(0,1.2)
    plt.plot(xs,y_pre)
    plt.scatter(xs,ys)
    plt.pause(0.01) #暂停0.01s

plt.show()

# 重新绘制散点图和预测曲线
# plt.scatter(xs,ys)
# y_pre = w*xs
# plt.plot(xs,y_pre)
# plt.show()

