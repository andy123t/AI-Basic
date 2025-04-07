# -*- coding: utf-8 -*-
# 2025-1-25 18:00:00
# Author: Andy

import dataset
from matplotlib import pyplot as plt
import numpy as np

xs,ys =dataset.get_beans(100)
# print(xs)
# print(ys)

# 配置图像
plt.title("Size-toxicity Function",fontsize=12) #设置图像名字
plt.xlabel("Bean Size") #设置横坐标的名字
plt.ylabel("Toxicity") #设置纵坐标的名字

plt.scatter(xs,ys)

# def sigmoid(z):
#     return 1/(1+np.exp(-z))

w = 0.1
b = 0.1
z = w*xs + b
a = 1/(1+np.exp(-z))

plt.plot(xs,a)
plt.show()

num = 5000
for _ in range(num):
    print("Epoch: "+str(_+1)+" of "+str(num))
    for i in range(100):
        x = xs[i]
        y = ys[i]
        
        # 对w和b求 (偏) 导
        z = w*x + b
        a = 1/(1+np.exp(-z))
        e = (y-a)**2
        
        deda = -2*(y-a)
        dadz = a*(1-a)
        dzdw = x
        dzdb = 1

        dedw = deda*dadz*dzdw
        dedb = deda*dadz*dzdb
        
        alpha = 0.05
        w = w - alpha*dedw
        b = b - alpha*dedb

    if _ %100 == 0:
        plt.clf() #清空窗口
        plt.scatter(xs,ys)
        z = w*xs + b
        a = 1/(1+np.exp(-z))

        plt.xlim(0,1)
        plt.ylim(0,1.2)
        plt.plot(xs,a)
        plt.pause(0.1) #暂停0.1s

