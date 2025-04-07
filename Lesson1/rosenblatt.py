# -*- coding: utf-8 -*-
# 2023-04-19 21:41:42
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

# Ronsenblatt model
for m in range(100):
    for i in range(100):
        x = xs[i]
        y = ys[i]
        y_pre = w*x
        e = y - y_pre
        alpha = 0.05
        w = w + alpha*e*x


# Predict result
y_pre = w * xs

print(y_pre)

plt.plot(xs,y_pre)

plt.show()


