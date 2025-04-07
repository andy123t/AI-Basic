# -*- coding: utf-8 -*-
# 2024-04-21 18:30:45
# Author: Andy

import dataset
from matplotlib import pyplot as plt
import numpy as np

# 从数据中获取随机豆豆
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

num = 5
for j in range(num):
    print("progress bar: "+str(j+1)+" of "+str(num))
    for i in range(n):
        x = xs[i]
        y = ys[i]
        # a=x^2
        # b=-2*x*y
        # c=y^2
        # 斜率k=2aw+b
        k = 2*(x**2)*w + (-2*x*y)
        alpha = 0.1
        w = w - alpha*k
        plt.clf() #清空窗口
        plt.scatter(xs,ys)
        y_pre = w*xs
        plt.xlim(0,1)
        plt.ylim(0,1.2)
        plt.plot(xs,y_pre)
        plt.pause(0.01) #暂停0.01s

plt.show()

# 重新绘制散点图和预测曲线
# plt.scatter(xs,ys)
# y_pre = w*xs
# plt.plot(xs,y_pre)
# plt.show()

