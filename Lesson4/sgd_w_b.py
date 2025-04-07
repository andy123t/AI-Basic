# -*- coding: utf-8 -*-
# 2025-1-20 19:05:00
# Author: Andy

import dataset
from matplotlib import pyplot as plt
import numpy as np

# 从数据中获取随机豆豆
m = 100

xs,ys = dataset.get_beans(m)

# 配置图像
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean Size") #设置横坐标的名字
plt.ylabel("Toxicity") #设置纵坐标的名字

plt.scatter(xs,ys)

w = 0.1
b = 0.01
y_pre = w*xs + b

plt.plot(xs,y_pre)

# plt.show()

# 训练次数
num = 100

for _ in range(num):
    print("Iteration: "+str(_+1)+" of "+str(num))
    for i in range(m):
        x = xs[i]
        y = ys[i]
        # 对w和b求 (偏) 导
        dw = 2*(x**2)*w + 2*x*b - 2*x*y
        db = 2*b + 2*x*w - 2*y
        alpha = 0.01
        w = w - alpha*dw
        b = b - alpha*db

    plt.clf() #清空窗口
    plt.scatter(xs,ys)
    y_pre = w*xs + b
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

