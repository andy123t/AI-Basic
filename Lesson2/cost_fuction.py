# -*- coding: utf-8 -*-
# 2024-04-20 14:19:00
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

plt.show()

es = (ys-y_pre)**2

sum_e = np.sum(es)

sum_e = (1/n)*sum_e

print(sum_e)

ws = np.arange(0,3,0.1)

es = []
for w in ws:
    y_pre = w*xs
    e = (1/n)*np.sum((ys-y_pre)**2)
    #print("w:"+str(w)+" e:"+str(e))
    es.append(e)

# 配置图像
plt.title("cost function", fontsize=12)
plt.xlabel("w") #设置横坐标的名字
plt.ylabel("e") #设置纵坐标的名字
plt.plot(ws,es)
plt.show()

w_min = np.sum(xs*ys)/np.sum(xs*xs)
print("e takes minimum value, w:"+str(w_min))

# 配置图像
plt.title("Size-Toxicity Function", fontsize=12)
plt.xlabel("Bean Size") #设置横坐标的名字
plt.ylabel("Toxicity") #设置纵坐标的名字

plt.scatter(xs,ys)

y_pre = w_min*xs

plt.plot(xs,y_pre)

plt.show()


