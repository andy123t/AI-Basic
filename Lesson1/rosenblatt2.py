# -*- coding: utf-8 -*-
# 2023-04-19 21:43:30
# Author: Andy

import dataset
from matplotlib import pyplot as plt

xs,ys = dataset.get_beans(100)

print(xs)
print(ys)

# 配置图像
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean Size") #设置横坐标的名字
plt.ylabel("Toxicity") #设置纵坐标的名字

plt.scatter(xs,ys)

# y=0.5*x
w = 0.5

# 迭代次数
n = 20

# Ronsenblatt model
for m in range(n):
    print("Epoch: "+str(m+1)+" of "+str(n))
    for i in range(100):
        x = xs[i]
        y = ys[i]
        y_pre = w*x
        e = y - y_pre
        alpha = 0.005
        w = w + alpha*e*x
    
    # Predict result
    plt.clf() #清空窗口
    plt.scatter(xs,ys)
    y_pre = w * xs
    print(y_pre)
    plt.plot(xs,y_pre,color='red')
    plt.show(block=False)  #防止窗口被阻塞
    plt.pause(0.1)  #给一点时间展示每次绘图

plt.show()  # 最后展示所有图形
