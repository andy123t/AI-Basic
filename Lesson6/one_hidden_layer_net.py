# -*- coding: utf-8 -*-
# 2025-2-12 20:05:28
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

def sigmoid(x):
    return 1/(1+np.exp(-x))

# 第一层的权重和偏置
w11_1 = np.random.rand()
w12_1 = np.random.rand()
b1_1 = np.random.rand()
b2_1 = np.random.rand()

# 第二层的权重换个偏置
w11_2 = np.random.rand()
w21_2 = np.random.rand()
b1_2 = np.random.rand()

# 前向传播定义函数
def forward_propgation(xs):
    z1_1 = w11_1*xs + b1_1
    a1_1 = sigmoid(z1_1)

    z2_1 = w12_1*xs + b2_1
    a2_1 = sigmoid(z2_1)

    z1_2 = w11_2*a1_1 + w21_2*a2_1 + b1_2
    a1_2 = sigmoid(z1_2)
    return a1_1,z1_1,a1_2,z1_2,a2_1,z2_1

a1_1,z1_1,a1_2,z1_2,a2_1,z2_1 = forward_propgation(xs)

plt.plot(xs,a1_2)

plt.show()

num = 5000
for _ in range(num):
    print("Epoch: "+str(_+1)+" of "+str(num))
    for i in range(100):
        x = xs[i]
        y = ys[i]
        
        # 反向传播
        a1_1,z1_1,a1_2,z1_2,a2_1,z2_1 = forward_propgation(x)
        e = (y - a2_1)**2
        
        # 最后一层
        deda1_2 = -2*(y-a1_2)
        da1_2dz1_2 = a1_2*(1-a1_2)
        dz1_2dw11_2 = a1_1
        dz1_2dw21_2 = a2_1
        dz1_2db1_2 = 1
        
        dedw11_2 = deda1_2*da1_2dz1_2*dz1_2dw11_2
        dedw21_2 = deda1_2*da1_2dz1_2*dz1_2dw21_2
        dedb1_2 = deda1_2*da1_2dz1_2*dz1_2db1_2

        # 倒数第二层
        dz1_2da1_1 = w11_2
        dz1_2da2_1 = w21_2
        da1_1dz1_1 = a1_1*(1-a1_1)
        da2_1dz2_1 = a2_1*(1-a2_1)

        dz1_1dw11_1 = x
        dz1_1db1_1 = 1
        dz2_1dw12_1 = x
        dz2_1db2_1 = 1

        dedw11_1 = deda1_2*da1_2dz1_2*dz1_2da1_1*da1_1dz1_1*dz1_1dw11_1
        dedw12_1 = deda1_2*da1_2dz1_2*dz1_2da2_1*da2_1dz2_1*dz2_1dw12_1

        dedb1_1 = deda1_2*da1_2dz1_2*dz1_2da1_1*da1_1dz1_1*dz1_1db1_1
        dedb2_1 = deda1_2*da1_2dz1_2*dz1_2da2_1*da2_1dz2_1*dz2_1db2_1

        alpha = 0.05

        w11_1 = w11_1 - alpha*dedw11_1
        w12_1 = w12_1 - alpha*dedw12_1
        w11_2= w11_2 - alpha*dedw11_2
        w21_2 = w21_2 - alpha*dedw21_2

        b1_1 = b1_1 - alpha*dedb1_1
        b2_1 = b2_1 - alpha*dedb2_1
        b1_2 = b1_2 - alpha*dedb1_2
        
    if _ %100 == 0:
        plt.clf() #清空窗口
        plt.scatter(xs,ys)
        a1_1,z1_1,a1_2,z1_2,a2_1,z2_1 = forward_propgation(xs)
        
        plt.xlim(0,2)
        plt.ylim(0,1.2)
        plt.plot(xs,a1_2)
        plt.pause(0.1) #暂停0.1s

