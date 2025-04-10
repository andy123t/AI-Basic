# -*- coding: utf-8 -*-
# 2025-4-2 18:15:24
# Author: Andy

# Windows 报错修复
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import dataset
import torch
import plot_utils

m = 100
X, Y = dataset.get_beans(m)
print(X)
print(Y)

plot_utils.show_scatter(X, Y)

# w1 = 0.1
# w2 = 0.2
W = np.array([0.1, 0.1])
# b = 0.1
B = np.array([0.1])

# 将 W 和 B 转换为 Tensor 类型，并开启梯度计算
W = torch.tensor(W, dtype=torch.float32, requires_grad=True)
B = torch.tensor(B, dtype=torch.float32, requires_grad=True)

print(W.shape)
print(B.shape)

def forward_propagation(X):
    # 避免使用 torch.tensor 产生警告
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    # z = w1*x1s + w2*x2s + b
    Z = torch.matmul(X, W) + B
    # a = 1 / (1 + torch.exp(-z))
    A = 1 / (1 + torch.exp(-Z))
    return A.detach().numpy()

plot_utils.show_scatter_surface(X, Y, forward_propagation)

alpha = 0.01
num = 1000
for _ in range(num):
    print(f"Epoch: {_+1}/{num}")
    for i in range(m):
        Xi = torch.from_numpy(X[i]).float()
        Yi = torch.tensor(Y[i], dtype=torch.float32)

        # 前向传播
        Z = torch.matmul(Xi, W) + B
        A = 1 / (1 + torch.exp(-Z))

        # 计算损失
        E = (Yi - A) ** 2

        # 手动计算梯度
        dEdA = -2 * (Yi - A)
        dAdZ = A * (1 - A)
        dZdW = Xi
        dZdB = torch.tensor(1.0, dtype=torch.float32)

        dE_dW = dEdA * dAdZ * dZdW
        dE_dB = dEdA * dAdZ * dZdB

        # 更新参数
        W = W - alpha * dE_dW
        B = B - alpha * dE_dB

        # 确保 W 和 B 仍然是可训练的张量
        W = W.detach().requires_grad_(True)
        B = B.detach().requires_grad_(True)

plot_utils.show_scatter_surface(X, Y, forward_propagation)

