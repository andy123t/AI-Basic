# -*- coding: utf-8 -*-
# 2025-4-10 18:40:05
# Author: Andy

# Fix error in Windows
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# import numpy as np
import dataset
import plot_utils
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 从数据中获取随机豆豆
m = 100
X, Y = dataset.get_beans(m)
print(X)
print(Y)
plot_utils.show_scatter(X, Y)

# 转换为 Tensor 类型
X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()

# 定义模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 8)  # 输入层到第一个隐藏层
        self.relu = nn.ReLU()  # ReLU 激活函数
        self.fc2 = nn.Linear(8, 8)  # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(8, 8)  # 第二个隐藏层到第三个隐藏层
        self.fc4 = nn.Linear(8, 1)  # 第三个隐藏层到输出层
        self.sigmoid = nn.Sigmoid()  # 输出层的激活函数

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

    def get_weights(self):
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = param.data
        return weights

model = SimpleNet()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

# 训练模型
epochs = 20000
batch_size = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs.squeeze(), Y)  # 调整维度以匹配
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        predicted_labels = (outputs.squeeze() >= 0.5).float()
        accuracy = (predicted_labels == Y).float().mean()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch: {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.2f}')

# 进行预测
# with torch.no_grad():
#     predictions = model(X)
#     predicted_labels = (predictions.squeeze() >= 0.5).float()
#     accuracy = (predicted_labels == Y).float().mean()
#     print(f'Final Accuracy: {accuracy.item():.4f}')

# 输出参数
weights = model.get_weights()
for name, param in weights.items():
    print(f'{name}: {param}')

plot_utils.show_scatter_surface(X, Y, model)

