# -*- coding: utf-8 -*-
# 2025-4-2 18:36:05
# Author: Andy

# Windows 报错修复
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import dataset
import plot_utils
import numpy as np

# 从数据中获取随机豆豆
m = 100
X, Y = dataset.get_beans4(m)
print(X)
print(Y)

plot_utils.show_scatter(X, Y)

# 初始化权重和偏置
W1 = np.random.rand(2, 2)
B1 = np.random.rand(1, 2)
W2 = np.random.rand(1, 2)
B2 = np.random.rand(1, 1)


def forward_propagation(X):
    """
    前向传播函数
    :param X: 输入数据
    :return: A2: 输出层的激活值, Z2: 输出层的线性组合值, A1: 隐藏层的激活值, Z1: 隐藏层的线性组合值
    """
    # 计算隐藏层的线性组合值 (m, 2)
    Z1 = np.dot(X, W1.T) + B1
    # 计算隐藏层的激活值 (m, 2)
    A1 = 1 / (1 + np.exp(-Z1))
    # 计算输出层的线性组合值 (m, 1)
    Z2 = np.dot(A1, W2.T) + B2
    # 计算输出层的激活值 (m, 1)
    A2 = 1 / (1 + np.exp(-Z2))
    return A2, Z2, A1, Z1


# 由于 plot_utils.show_scatter_surface 期望的函数返回一个 numpy 数组，这里进行适配
def adapted_forward_propagation(X):
    A2, _, _, _ = forward_propagation(X)
    return A2


plot_utils.show_scatter_surface(X, Y, adapted_forward_propagation)

epochs = 5000
for epoch in range(epochs):
    # print("Epoch: "+str(_+1)+" of "+str(num))
    for i in range(m):
        Xi = X[i].reshape(1, -1)  # 确保 Xi 是二维数组 (1, 2)
        Yi = Y[i].reshape(1, -1)  # 确保 Yi 是二维数组 (1, 1)

        A2, Z2, A1, Z1 = forward_propagation(Xi)

        E = (Yi - A2) ** 2

        # 计算梯度
        dEdA2 = -2 * (Yi - A2)  # (1, 1)
        dEdZ2 = dEdA2 * A2 * (1 - A2)  # (1, 1)
        dEdW2 = np.dot(dEdZ2.T, A1)  # (1, 2)
        dEdB2 = dEdZ2  # (1, 1)
        dEdA1 = np.dot(dEdZ2, W2)  # (1, 2)
        dEdZ1 = dEdA1 * A1 * (1 - A1)  # (1, 2)
        dEdW1 = np.dot(dEdZ1.T, Xi)  # (2, 2)
        dEdB1 = dEdZ1  # (1, 2)

        alpha = 0.05
        # 更新权重和偏置
        W2 = W2 - alpha * dEdW2
        B2 = B2 - alpha * dEdB2
        W1 = W1 - alpha * dEdW1
        B1 = B1 - alpha * dEdB1

    # 计算准确率
    A2, _, _, _ = forward_propagation(X)
    A2 = np.around(A2)  # 四舍五入取出 0.5 分割线左右的分类结果
    A2 = A2.reshape(-1)
    accuracy = np.mean(np.equal(A2, Y))
    print(f"Epoch: {epoch+1}, 准确率: {accuracy}")

plot_utils.show_scatter_surface(X, Y, adapted_forward_propagation)

